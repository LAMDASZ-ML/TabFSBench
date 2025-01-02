from model.methods.base import Method

class DCN2Method(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy == 'indices')

    def construct_model(self, model_config = None):
        if model_config is None:
            model_config = self.args.config['model']
        from model.models.dcn2 import DCNv2
        self.model = DCNv2(
                categories=self.categories,
                d_in=self.d_in,
                d_out=self.d_out,
                **model_config
        ).to(self.args.device)
        self.model.double()