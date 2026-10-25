"""Object registration must preserve handedness, camera rays and source motion."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.lamp.prepare_object_viewer import register_without_camera, registered_camera
from scripts.lamp.prepare_lamp_viewer import rigid_deltas
from scripts.motion.bind_lego_parts import transform


class ObjectPreparationTests(unittest.TestCase):
    def test_gray_backdrop_preserves_enclosed_and_dark_parts(self):
        from scripts.viewer.prepare_reference_videos import white_rgb_preview
        rgb = np.full((30,30,3), 69, dtype=np.uint8)
        rgb[7:23,7:23] = 20
        rgb[12:18,12:18] = 69
        result = white_rgb_preview(rgb, background_rgb=[69,69,69])
        np.testing.assert_array_equal(result[0], 255)
        np.testing.assert_array_equal(result[7:23,7:23], rgb[7:23,7:23])
        with self.assertRaisesRegex(ValueError, 'border'):
            white_rgb_preview(rgb, background_rgb=[0,0,0])

    def test_camera_free_registration_recovers_asymmetric_geometry(self):
        rng = np.random.default_rng(14)
        source = np.concatenate([rng.normal(size=(240, 3))*[.8,.2,.1],
                                 rng.normal(size=(150, 3))*[.2,.1,.3]+[.7,.5,.1]])
        expected = np.eye(4)
        expected[:3, :3] = 1.4*Rotation.from_euler('xyz', [61,23,-45], degrees=True).as_matrix()
        expected[:3, 3] = [.3,-.6,.9]
        target = transform(source, expected)
        actual, report = register_without_camera(source, target)
        self.assertGreater(np.linalg.det(actual[:3, :3]), 0)
        self.assertLess(np.sqrt(np.mean((transform(source, actual)-target)**2)), .015)
        self.assertLess(report['bidirectional_distance'], .03)

    def test_registered_camera_preserves_pixels_and_rigid_deltas(self):
        source_pose = np.eye(4); source_pose[2, 3] = 5
        registration = np.eye(4)
        registration[:3, :3] = 2*Rotation.from_euler('x', -90, degrees=True).as_matrix()
        registration[:3, 3] = [.2,.3,.4]
        target = transform(np.zeros((1,3)), registration)[0]
        camera = registered_camera(source_pose, registration, target, 40)
        pose = np.asarray(camera['pose'])
        np.testing.assert_allclose(pose[:3,:3].T @ pose[:3,:3], np.eye(3), atol=1e-12)
        point = np.array([[.3,.2,0]])
        before = transform(point, np.linalg.inv(source_pose))
        after = transform(transform(point, registration), np.linalg.inv(pose))
        np.testing.assert_allclose(before[:,:2]/before[:,2:], after[:,:2]/after[:,2:])
        matrices = np.tile(np.eye(4), (2,1,1,1))
        matrices[1,0,:3,:3] = Rotation.from_euler('z', 20, degrees=True).as_matrix()
        delta = rigid_deltas(matrices, registration)
        np.testing.assert_allclose(delta[0,0], np.eye(4))
        moved = transform(transform(point, registration), delta[1,0])
        np.testing.assert_allclose(moved, transform(transform(point, matrices[1,0]), registration))


if __name__ == '__main__':
    unittest.main()
