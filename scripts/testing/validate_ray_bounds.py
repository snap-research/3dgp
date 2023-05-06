"""
Validates that the rays which are casted properly hit the [-1, 1]^3 cube
"""

import sys; sys.path.append('.')
import argparse
from src.training.tri_plane_renderer import validate_image_plane
from src.training.rendering_utils import validate_frustum


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--radius', type=float, help='Camera distance from the origin')
    parser.add_argument('--fov', type=float, help='Field of view (in degrees)')
    parser.add_argument('--scale', type=float, default=1.0, help='The additional scaling of the [-1, 1] cube (if any)')
    parser.add_argument('--near', type=float, help='Near plane position (if validating the frustum)')
    parser.add_argument('--far', type=float, help='Far plane position (if validating the frustum)')
    parser.add_argument('--use_full_box', action='store_true', help='Should we use the full box?')
    parser.add_argument('--step', type=float, default=1e-2, help='Step size when sampling the points')
    parser.add_argument('--device', type=str, default='cpu', help='`cpu` or `cuda` device?')
    args = parser.parse_args()

    if args.use_full_box:
        is_valid = validate_image_plane(
            radius=args.radius,
            fov=args.fov,
            scale=args.scale,
            step=args.step,
            device=args.device,
        )
    else:
        is_valid = validate_frustum(
           radius=args.radius,
            near=args.near,
            far=args.far,
            fov=args.fov,
            scale=args.scale,
            step=args.step,
            device=args.device,
            verbose=True,
        )
    print('Valid?', is_valid)
