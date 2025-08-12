import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K

# BCE loss for the discriminator
def d_bce_loss():
    def loss(y_true, y_pred):
        d_bce_loss = binary_crossentropy(y_true, y_pred)
        return d_bce_loss
    return loss

# trajLoss for the generator - adapted for neutron trajectories
def neutron_trajLoss(real_traj, gen_traj):
    def loss(y_true, y_pred):
        # Extract mask for proper loss calculation
        mask = real_traj[2]  # mask is the 3rd input
        traj_length = K.sum(mask, axis=1)
        
        # Adversarial loss (BCE)
        bce_loss = binary_crossentropy(y_true, y_pred)
        
        # 3D coordinate MSE loss (masked)
        coords_diff = gen_traj[0] - real_traj[0]  # coordinates difference
        coords_squared_diff = tf.multiply(coords_diff, coords_diff)
        
        # Apply mask to ignore padded positions
        # Repeat mask for all 3 coordinates (x, y, z)
        mask_3d = tf.concat([mask for _ in range(3)], axis=2)
        masked_coords_loss = tf.multiply(coords_squared_diff, mask_3d)
        
        # Sum over coordinates and sequence, then average by trajectory length
        coords_mse_full = K.sum(K.sum(masked_coords_loss, axis=1), axis=1, keepdims=True)
        coords_mse = K.sum(tf.math.divide(coords_mse_full, traj_length * 3))  # Divide by 3 for x,y,z
        
        # Arc length cross-entropy loss (masked)
        ce_arc = tf.nn.softmax_cross_entropy_with_logits(
            labels=real_traj[1],  # real arc length (one-hot)
            logits=gen_traj[1]    # generated arc length (logits)
        )
        
        # Apply mask to arc length loss
        mask_1d = K.sum(mask, axis=2)  # Reduce mask dimension
        ce_arc_masked = tf.multiply(ce_arc, mask_1d)
        ce_arc_mean = K.sum(tf.math.divide(ce_arc_masked, K.sum(mask, axis=1)))
        
        # Loss weights
        p_bce = 1.0          # Adversarial loss weight
        p_coords = 10.0      # Coordinate reconstruction weight
        p_arc = 1.0          # Arc length consistency weight
        
        total_loss = (bce_loss * p_bce + 
                     coords_mse * p_coords + 
                     ce_arc_mean * p_arc)
        
        return total_loss
    
    return loss
