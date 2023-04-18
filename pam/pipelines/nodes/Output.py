from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface
import os


class InputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)


class OutputSpec(TraitedSpec):
    out_file = File(genfile=True)


class Output(BaseInterface):
    input_spec = InputSpec
    output_spec = OutputSpec

    def _run_interface(self, runtime):
        from pam.dataset.uploader import upload_blob_to_container
        out_put = self.inputs.in_file

        upload_blob_to_container(out_put)
        return runtime

    def _list_outputs(self):
        return {'out_file': ''}

    def _gen_filename(self, name):
        if name == 'out_file':
            return os.path.abspath(self.inputs.out_file)
        return None