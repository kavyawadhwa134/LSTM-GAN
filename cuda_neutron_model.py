import tensorflow as tf
import numpy as np
import random

# Enable GPU memory growth to prevent memory issues
def configure_gpu():
    """Configure GPU settings for optimal performance"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for each GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set the first GPU as the default
            tf.config.set_visible_devices(gpus[0], 'GPU')
            
            # Enable mixed precision for better performance on modern GPUs
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            print(f"GPU configuration completed. Available GPUs: {len(gpus)}")
            print(f"Mixed precision enabled: {policy.name}")
            
            return True
        except RuntimeError as e:
            print(f"GPU configuration failed: {e}")
            return False
    else:
        print("No GPUs found. Running on CPU.")
        return False

# Configure GPU on import
GPU_AVAILABLE = configure_gpu()

random.seed(2020)
np.random.seed(2020)
tf.random.set_seed(2020)

from tensorflow.keras.layers import Input, Add, Average, Dense, LSTM, Lambda, TimeDistributed, Concatenate, Embedding
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.regularizers import l1

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

from neutron_losses import d_bce_loss, neutron_trajLoss

class CUDA_Neutron_LSTM_TrajGAN():
    def __init__(self, latent_dim, max_length, coordinate_bounds, use_mixed_precision=True):
        self.latent_dim = latent_dim
        self.max_length = max_length
        
        # For neutron trajectories: 3D coordinates, arc_length bins, mask
        self.keys = ['coordinates', 'arc_length', 'mask']
        self.vocab_size = {"coordinates": 3, "arc_length": 10, "mask": 1}
        
        self.coordinate_bounds = coordinate_bounds  # For denormalization
        self.use_mixed_precision = use_mixed_precision and GPU_AVAILABLE
        
        self.x_train = None
        
        # Device strategy for multi-GPU if available
        self.strategy = self._get_strategy()
        
        with self.strategy.scope():
            # Define the optimizer with GPU-optimized settings
            self.optimizer = self._get_optimizer()

            # Build the trajectory generator
            self.generator = self.build_generator()

            # The trajectory generator takes real trajectories and noise as inputs
            noise = Input(shape=(self.latent_dim,), name='input_noise')
            inputs = []
            
            # Coordinates input
            coords_input = Input(shape=(self.max_length, 3), name='input_coordinates')
            inputs.append(coords_input)
            
            # Arc length input (one-hot encoded)
            arc_input = Input(shape=(self.max_length, 10), name='input_arc_length')
            inputs.append(arc_input)
            
            # Mask input
            mask_input = Input(shape=(self.max_length, 1), name='input_mask')
            inputs.append(mask_input)
            
            inputs.append(noise)
            
            # The trajectory generator generates synthetic trajectories
            gen_trajs = self.generator(inputs)
            
            # Build and compile the discriminator
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(
                loss=d_bce_loss(), 
                optimizer=self.optimizer, 
                metrics=['accuracy']
            )

            # The combined model only trains the trajectory generator
            self.discriminator.trainable = False

            # The discriminator takes generated trajectories as input and makes predictions
            pred = self.discriminator(gen_trajs[:3])  # Don't pass mask to discriminator

            # The combined model (combining the generator and the discriminator)
            self.combined = Model(inputs, pred)
            self.combined.compile(
                loss=neutron_trajLoss(inputs, gen_trajs), 
                optimizer=self.optimizer
            )
        
        # Save model architectures
        self._save_model_architectures()
        
        print(f"CUDA Neutron GAN initialized with strategy: {type(self.strategy).__name__}")
        print(f"Mixed precision: {self.use_mixed_precision}")
    
    def _get_strategy(self):
        """Get the appropriate distribution strategy"""
        gpus = tf.config.list_physical_devices('GPU')
        
        if len(gpus) > 1:
            # Multi-GPU strategy
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using MirroredStrategy with {len(gpus)} GPUs")
        elif len(gpus) == 1:
            # Single GPU strategy
            strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
            print("Using OneDeviceStrategy with 1 GPU")
        else:
            # CPU fallback
            strategy = tf.distribute.get_strategy()  # Default strategy
            print("Using default strategy (CPU)")
        
        return strategy
    
    def _get_optimizer(self):
        """Get GPU-optimized optimizer"""
        learning_rate = 0.001
        
        if GPU_AVAILABLE:
            # Use Adam with GPU-optimized settings
            optimizer = Adam(
                learning_rate=learning_rate,
                beta_1=0.5,
                beta_2=0.999,
                epsilon=1e-7,  # Smaller epsilon for better GPU performance
                amsgrad=False
            )
        else:
            # Standard optimizer for CPU
            optimizer = Adam(learning_rate, 0.5)
        
        # Wrap with mixed precision if enabled
        if self.use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        return optimizer
    
    def _save_model_architectures(self):
        """Save model architectures with error handling"""
        import os
        os.makedirs("params", exist_ok=True)
        
        try:
            C_model_json = self.combined.to_json()
            with open("params/cuda_neutron_C_model.json", "w") as json_file:
                json_file.write(C_model_json)
                
            G_model_json = self.generator.to_json()
            with open("params/cuda_neutron_G_model.json", "w") as json_file:
                json_file.write(G_model_json)
            
            D_model_json = self.discriminator.to_json()
            with open("params/cuda_neutron_D_model.json", "w") as json_file:
                json_file.write(D_model_json)
        except Exception as e:
            print(f"Warning: Could not save model architectures: {e}")

    def build_discriminator(self):
        """Build GPU-optimized discriminator"""
        
        # Input layers
        coords_input = Input(shape=(self.max_length, 3), name='disc_input_coordinates')
        arc_input = Input(shape=(self.max_length, 10), name='disc_input_arc_length')
        mask_input = Input(shape=(self.max_length, 1), name='disc_input_mask')
        
        # Embedding layers with GPU-optimized activations
        coords_unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(coords_input)
        coords_dense = Dense(
            units=64, 
            use_bias=True, 
            activation='relu', 
            kernel_initializer=he_uniform(seed=1),
            dtype='float32' if not self.use_mixed_precision else 'mixed_float16',
            name='emb_coordinates'
        )
        coords_embedded = [coords_dense(x) for x in coords_unstacked]
        coords_embedded = Lambda(lambda x: tf.stack(x, axis=1))(coords_embedded)
        
        arc_unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(arc_input)
        arc_dense = Dense(
            units=32, 
            use_bias=True, 
            activation='relu',
            kernel_initializer=he_uniform(seed=1),
            dtype='float32' if not self.use_mixed_precision else 'mixed_float16',
            name='emb_arc_length'
        )
        arc_embedded = [arc_dense(x) for x in arc_unstacked]
        arc_embedded = Lambda(lambda x: tf.stack(x, axis=1))(arc_embedded)
        
        # Feature fusion with batch normalization for GPU optimization
        concat_input = Concatenate(axis=2)([coords_embedded, arc_embedded])
        
        # Apply mask to ignore padded positions
        masked_input = Lambda(lambda x: x[0] * x[1])([concat_input, mask_input])
        
        # Add batch normalization for better GPU training
        masked_input = tf.keras.layers.BatchNormalization()(masked_input)
        
        unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(masked_input)
        fusion_dense = Dense(
            units=100, 
            use_bias=True, 
            activation='relu', 
            kernel_initializer=he_uniform(seed=1),
            dtype='float32' if not self.use_mixed_precision else 'mixed_float16',
            name='emb_trajpoint'
        )
        fused_outputs = [fusion_dense(x) for x in unstacked]
        emb_traj = Lambda(lambda x: tf.stack(x, axis=1))(fused_outputs)
        
        # LSTM layer optimized for GPU (many-to-one)
        lstm_cell = LSTM(
            units=100, 
            recurrent_regularizer=l1(0.02),
            recurrent_activation='sigmoid',  # Better for GPU
            unroll=False,  # Let TensorFlow optimize
            use_bias=True
        )(emb_traj)
        
        # Output with explicit float32 for mixed precision
        sigmoid = Dense(1, activation='sigmoid', dtype='float32')(lstm_cell)

        return Model(inputs=[coords_input, arc_input, mask_input], outputs=sigmoid)

    def build_generator(self):
        """Build GPU-optimized generator"""
        
        # Input layers
        coords_input = Input(shape=(self.max_length, 3), name='gen_input_coordinates')
        arc_input = Input(shape=(self.max_length, 10), name='gen_input_arc_length')
        mask_input = Input(shape=(self.max_length, 1), name='gen_input_mask')
        noise = Input(shape=(self.latent_dim,), name='gen_input_noise')
        
        # Embedding layers with GPU optimizations
        coords_unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(coords_input)
        coords_dense = Dense(
            units=64, 
            activation='relu', 
            use_bias=True, 
            kernel_initializer=he_uniform(seed=1),
            dtype='float32' if not self.use_mixed_precision else 'mixed_float16',
            name='gen_emb_coordinates'
        )
        coords_embedded = [coords_dense(x) for x in coords_unstacked]
        coords_embedded = Lambda(lambda x: tf.stack(x, axis=1))(coords_embedded)
        
        arc_unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(arc_input)
        arc_dense = Dense(
            units=32, 
            activation='relu', 
            use_bias=True,
            kernel_initializer=he_uniform(seed=1),
            dtype='float32' if not self.use_mixed_precision else 'mixed_float16',
            name='gen_emb_arc_length'
        )
        arc_embedded = [arc_dense(x) for x in arc_unstacked]
        arc_embedded = Lambda(lambda x: tf.stack(x, axis=1))(arc_embedded)
        
        # Feature fusion with noise and batch normalization
        concat_input = Concatenate(axis=2)([coords_embedded, arc_embedded])
        concat_input = tf.keras.layers.BatchNormalization()(concat_input)
        
        unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(concat_input)
        fusion_dense = Dense(
            units=100, 
            use_bias=True, 
            activation='relu',
            kernel_initializer=he_uniform(seed=1),
            dtype='float32' if not self.use_mixed_precision else 'mixed_float16',
            name='gen_emb_trajpoint'
        )
        fused_outputs = [fusion_dense(Concatenate(axis=1)([x, noise])) for x in unstacked]
        emb_traj = Lambda(lambda x: tf.stack(x, axis=1))(fused_outputs)
        
        # LSTM layer optimized for GPU (many-to-many)
        lstm_cell = LSTM(
            units=100,
            return_sequences=True,
            recurrent_regularizer=l1(0.02),
            recurrent_activation='sigmoid',  # Better for GPU
            unroll=False,  # Let TensorFlow optimize
            use_bias=True
        )(emb_traj)
        
        # Add dropout for better generalization
        lstm_cell = tf.keras.layers.Dropout(0.1)(lstm_cell)
        
        # Outputs with explicit dtypes for mixed precision
        # 3D coordinates output
        coords_output = TimeDistributed(
            Dense(3, activation='tanh', dtype='float32'), 
            name='output_coordinates'
        )(lstm_cell)
        
        # Arc length output (10 classes)
        arc_output = TimeDistributed(
            Dense(10, activation='softmax', dtype='float32'), 
            name='output_arc_length'
        )(lstm_cell)
        
        # Mask output (pass through)
        mask_output = Lambda(lambda x: x, name='output_mask')(mask_input)
                
        return Model(
            inputs=[coords_input, arc_input, mask_input, noise], 
            outputs=[coords_output, arc_output, mask_output]
        )

    @tf.function
    def train_step(self, real_trajs, batch_size):
        """Optimized training step with tf.function for GPU acceleration"""
        
        # Ground truth labels
        real_bc = tf.ones((batch_size, 1), dtype=tf.float32)
        syn_bc = tf.zeros((batch_size, 1), dtype=tf.float32)
        
        # Generate noise
        noise = tf.random.normal((batch_size, self.latent_dim), dtype=tf.float32)
        
        # Prepare inputs
        real_trajs_with_noise = real_trajs + [noise]
        
        # Generate synthetic trajectories
        gen_trajs_bc = self.generator(real_trajs_with_noise, training=True)
        
        # Train discriminator
        with tf.GradientTape() as disc_tape:
            # Real predictions
            real_pred = self.discriminator(real_trajs[:3], training=True)
            real_loss = tf.keras.losses.binary_crossentropy(real_bc, real_pred)
            
            # Fake predictions
            fake_pred = self.discriminator(gen_trajs_bc[:3], training=True)
            fake_loss = tf.keras.losses.binary_crossentropy(syn_bc, fake_pred)
            
            # Total discriminator loss
            d_loss = 0.5 * (tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss))
            
            # Scale loss for mixed precision
            if self.use_mixed_precision:
                d_loss = self.optimizer.get_scaled_loss(d_loss)
        
        # Apply discriminator gradients
        d_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        if self.use_mixed_precision:
            d_gradients = self.optimizer.get_unscaled_gradients(d_gradients)
        self.optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Train generator
        with tf.GradientTape() as gen_tape:
            # Generate new noise for generator training
            noise = tf.random.normal((batch_size, self.latent_dim), dtype=tf.float32)
            real_trajs_with_noise[3] = noise
            
            # Generator loss
            g_loss = self.combined(real_trajs_with_noise, training=True)
            g_loss = tf.keras.losses.binary_crossentropy(real_bc, g_loss)
            g_loss = tf.reduce_mean(g_loss)
            
            # Scale loss for mixed precision
            if self.use_mixed_precision:
                g_loss = self.optimizer.get_scaled_loss(g_loss)
        
        # Apply generator gradients
        g_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        if self.use_mixed_precision:
            g_gradients = self.optimizer.get_unscaled_gradients(g_gradients)
        self.optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        # Return unscaled losses for logging
        if self.use_mixed_precision:
            d_loss = d_loss / self.optimizer.loss_scale
            g_loss = g_loss / self.optimizer.loss_scale
        
        return d_loss, g_loss

    def train(self, epochs=200, batch_size=32, sample_interval=10):
        """GPU-accelerated training with optimizations"""
        
        # Load training data
        train_data = np.load('data/neutron_train_final.npz', allow_pickle=True)
        x_train = [
            train_data['coordinates'].astype(np.float32),
            train_data['arc_length'].astype(np.float32), 
            train_data['mask'].astype(np.float32)
        ]
        self.x_train = x_train
        
        print(f"Training data shapes: {[arr.shape for arr in x_train]}")
        print(f"Training on device: {'/GPU:0' if GPU_AVAILABLE else '/CPU:0'}")
        
        # Convert to TensorFlow datasets for better GPU utilization
        dataset = tf.data.Dataset.from_tensor_slices(tuple(x_train))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Training loop with GPU optimizations
        for epoch in range(1, epochs + 1):
            
            # Use tf.data for better GPU memory management
            for batch_data in dataset.take(1):  # Take one random batch
                # Convert to list format expected by train_step
                real_trajs = [batch_data[0], batch_data[1], batch_data[2]]
                
                # Perform optimized training step
                d_loss, g_loss = self.train_step(real_trajs, batch_size)
                
                break  # Only use one batch per epoch (following original logic)
            
            # Convert tensors to numpy for printing
            d_loss_val = float(d_loss.numpy())
            g_loss_val = float(g_loss.numpy())
            
            print(f"[{epoch}/{epochs}] D Loss: {d_loss_val:.6f} | G Loss: {g_loss_val:.6f}")
            
            # Save checkpoints
            if epoch % sample_interval == 0:
                self.save_checkpoint(epoch)
                print('Model params saved to the disk.')
    
    def save_checkpoint(self, epoch):
        """Save model checkpoints with error handling"""
        import os
        os.makedirs("training_params", exist_ok=True)
        
        try:
            self.combined.save_weights(f"training_params/cuda_neutron_C_model_{epoch}.h5")
            self.generator.save_weights(f"training_params/cuda_neutron_G_model_{epoch}.h5")
            self.discriminator.save_weights(f"training_params/cuda_neutron_D_model_{epoch}.h5")
            print("Training Params Saved")
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")
    
    def load_checkpoint(self, epoch):
        """Load model checkpoints"""
        try:
            self.generator.load_weights(f"training_params/cuda_neutron_G_model_{epoch}.h5")
            self.discriminator.load_weights(f"training_params/cuda_neutron_D_model_{epoch}.h5")
            self.combined.load_weights(f"training_params/cuda_neutron_C_model_{epoch}.h5")
            print(f"Loaded checkpoint from epoch {epoch}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
    
    def get_device_info(self):
        """Get information about available devices"""
        gpus = tf.config.list_physical_devices('GPU')
        cpus = tf.config.list_physical_devices('CPU')
        
        print(f"Available GPUs: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
        
        print(f"Available CPUs: {len(cpus)}")
        print(f"Mixed precision enabled: {self.use_mixed_precision}")
        print(f"Current strategy: {type(self.strategy).__name__}")
