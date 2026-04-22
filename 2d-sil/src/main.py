
import argparse
from track import Track, generate_track
from renderer import TrackRenderer
from logger import logger


def main():
    
    parser = argparse.ArgumentParser(description="Ghost-P1 2D SIL")
    parser.add_argument("--seed", type=int, default=69, help="Track generation seed")
    parser.add_argument("--samples", type=int, default=500, help="Number of track sample points")
    parser.add_argument("--width", type=float, default=5.0, help="Track width in meters")
    args = parser.parse_args()

    track = generate_track(seed=args.seed, num_samples=args.samples, track_width=args.width)
    
    if not track:
        logger.error("Could not generate track")
        return

    logger.info(f"Track generated: {track.total_length:.1f}m, {track.num_points} points")

    renderer = TrackRenderer(track)
    renderer.plot_track()

if __name__ == "__main__":
    main()