# visualize.py
import pyvista as pv
import numpy as np

def plot_tracks(real_tracks, generated, xyz_min, xyz_max):
    plotter = pv.Plotter()
    plotter.set_background('black')

    for track in real_tracks:
        poly = pv.Spline(track, 100)
        plotter.add_mesh(poly, color="cyan", line_width=2, opacity=0.5)

    poly_gen = pv.Spline(generated, 100)
    plotter.add_mesh(poly_gen, color="yellow", line_width=5, label="Generated")

    plotter.add_legend()
    plotter.show_grid()
    plotter.show_axes()
    plotter.show()

# Load and plot
if __name__ == "__main__":
    from load_data import load_and_split_tracks, preprocess_tracks
    raw_tracks = load_and_split_tracks()
    _, _, xyz_min, xyz_max = preprocess_tracks(raw_tracks)
    real_world = [t * (xyz_max - xyz_min) + xyz_min for t in raw_tracks]
    generated = np.loadtxt('generated_track.csv', delimiter=',', skiprows=1)
    plot_tracks(real_world, generated, xyz_min, xyz_max)