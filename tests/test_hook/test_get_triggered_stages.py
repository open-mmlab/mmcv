from mmcv.runner import Hook


def test_get_triggered_stages():

    class ToyHook(Hook):
        # test normal stage
        def before_run(self, runner):
            pass

        # test the method mapped to multi stages.
        def after_epoch(self, runner):
            pass

    hook = ToyHook()
    # stages output have order, so here is list instead of set.
    expected_stages = ['before_run', 'after_train_epoch', 'after_val_epoch']
    assert hook.get_triggered_stages() == expected_stages
