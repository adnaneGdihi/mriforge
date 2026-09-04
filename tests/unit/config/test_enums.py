from spectramr.config.schemas.enums import (
    GANLossType,
    LogLevel,
    LRScheduler,
    NoiseSchedule,
    OptimizerType,
    TrainingMode,
)


class TestEnums:
    def test_training_mode(self):
        assert TrainingMode.GAN == "gan"
        assert TrainingMode.DIFFUSION == "diffusion"

    def test_gan_loss_type(self):
        assert GANLossType.WGAN_GP == "wgan-gp"
        assert GANLossType.BCE == "bce"

    def test_lr_scheduler(self):
        assert LRScheduler.COSINE == "cosine"
        assert LRScheduler.STEP == "step"

    def test_noise_schedule(self):
        assert NoiseSchedule.LINEAR == "linear"
        assert NoiseSchedule.COSINE == "cosine"

    def test_optimizer_type(self):
        assert OptimizerType.ADAM == "adam"
        assert OptimizerType.SGD == "sgd"

    def test_log_level(self):
        assert LogLevel.INFO == "info"
        assert LogLevel.DEBUG == "debug"
