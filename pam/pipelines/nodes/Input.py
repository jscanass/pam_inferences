from nipype.interfaces.base import TraitDictObject, Directory, BaseInterfaceInputSpec, Str, File, TraitedSpec, traits, BaseInterface
import os


class InputSpec(BaseInterfaceInputSpec):
    site = Str(mandatory=True)


class OutputSpec(TraitedSpec):
    folder_output = Directory()


class Input(BaseInterface):
    input_spec = InputSpec
    output_spec = OutputSpec

    def _run_interface(self, runtime):

        from pam.dataset.searcher import get_records
        site = self.inputs.site

        output = get_records(site, path_home=False, download=True)

        return runtime

    def _list_outputs(self):
        return {'folder_output': os.path.abspath('.chorus') + '/' + self.inputs.site + '/records'}

    def _gen_filename(self, name):
        if name == 'folder_output':
            return os.path.abspath('chorus')
        return None