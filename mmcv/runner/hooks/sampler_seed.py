from .hook import Hook


class DistSamplerSeedHook(Hook):

    def before_epoch(self, runner):
        runner.data_loader.sampler.set_epoch(runner.epoch)
