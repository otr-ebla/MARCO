import unittest

from src.train_bosco import policy_checkpoint_score


class CheckpointSelectionTest(unittest.TestCase):
    def test_e2e_keeps_coverage_when_contacts_fall(self):
        # Captured fine-tuning updates 1510 and 1530.
        earlier = policy_checkpoint_score('end-to-end', 0., .7733, 81.48, -9259.427)
        later = policy_checkpoint_score('end-to-end', 0., .7595, 78.42, -8208.829)
        self.assertGreater(earlier, later)

    def test_guided_preserves_existing_contact_priority(self):
        earlier = policy_checkpoint_score('guided', 0., .7733, 81.48, -9259.427)
        later = policy_checkpoint_score('guided', 0., .7595, 78.42, -8208.829)
        self.assertGreater(later, earlier)

    def test_completion_remains_first(self):
        completed = policy_checkpoint_score('end-to-end', .1, .7, 100., -1000.)
        incomplete = policy_checkpoint_score('end-to-end', 0., .9, 0., 0.)
        self.assertGreater(completed, incomplete)
