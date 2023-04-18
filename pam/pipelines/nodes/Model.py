from nipype.interfaces.base import DictStrStr, Str, Directory, BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface
import os


class ModelInputSpec(BaseInterfaceInputSpec):
    model_path = Str(mandatory=True)
    model_metadata = Str(mandatory=True)
    folder_input = Directory(mandatory=True)

class ModelOutputSpec(TraitedSpec):
    out_put = File(genfile=True)


class Model(BaseInterface):
    input_spec = ModelInputSpec
    output_spec = ModelOutputSpec

    def _run_interface(self, runtime):
        import ast
        from core.inferences import run_inferences

        data_path = self.inputs.folder_input
        model_path = self.inputs.model_path
        model_metadata = ast.literal_eval(self.inputs.model_metadata)

        run_inferences(data_path=data_path, model_path=model_path, model_metadata=model_metadata)

        return runtime

    def _list_outputs(self):
        return {'out_put': os.path.abspath(self.inputs.folder_input.split('chorus')[-1].split('records')[0].replace('/','') + '_' + 'inferences_torch.parquet.gzip')}

    def _gen_filename(self, name):
        if name == 'out_put':
            return os.path.abspath(self.inputs.out_file)
        return None