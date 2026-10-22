"""Pure metadata guards for independently reloaded event-only audits."""
import unittest

from scripts.events.audit_event_only_checkpoint import checkpoint_observation_settings, validate_timeline


class CheckpointAuditGuards(unittest.TestCase):
    def payload(self):
        return {'input_provenance': {'g0_time': 0., 'time_scale': 1.},
                'training_summary': {'thresholds': [.2, .2], 'event_background': .048,
                                     'log_intensity_floor': .0001}}

    def test_saved_observation_model_is_preserved(self):
        settings = checkpoint_observation_settings(self.payload())
        self.assertEqual(settings['event_background'], .048)
        self.assertEqual(settings['log_intensity_floor'], .0001)
        self.assertEqual(settings['explicit_legacy_fallbacks'], [])

    def test_missing_radiometry_defaults_are_explicit(self):
        payload = self.payload()
        del payload['training_summary']['event_background']
        del payload['training_summary']['log_intensity_floor']
        settings = checkpoint_observation_settings(payload)
        self.assertEqual(settings['event_background'], 0.)
        self.assertEqual(settings['log_intensity_floor'], .01)
        self.assertEqual(len(settings['explicit_legacy_fallbacks']), 2)

    def test_missing_contrasts_or_clock_are_not_guessed(self):
        for section, field in (('training_summary', 'thresholds'),
                               ('input_provenance', 'g0_time'), ('input_provenance', 'time_scale')):
            payload = self.payload()
            del payload[section][field]
            with self.assertRaises(ValueError):
                checkpoint_observation_settings(payload)

    def test_invalid_radiometry_is_rejected(self):
        for field, value in (('event_background', -1), ('event_background', float('nan')),
                             ('log_intensity_floor', 0), ('log_intensity_floor', True)):
            payload = self.payload()
            payload['training_summary'][field] = value
            with self.assertRaises(ValueError):
                checkpoint_observation_settings(payload)

    def test_exact_full_timeline_is_required(self):
        self.assertEqual(validate_timeline([0., .025, .05, .075, .1], [0., .1])[0][-1], .1)
        for endpoints, knots in (([0., .1], [0., .2]), ([0., .2], [0., .1]),
                                  ([.01, .1], [0., .1]), ([0., .1, .1], [0., .1])):
            with self.assertRaises(ValueError):
                validate_timeline(endpoints, knots)


if __name__ == '__main__':
    unittest.main()
