import os
import nipype
from os.path import join as opj
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype import Node
from pam.pipelines.nodes.Input import Input
from pam.pipelines.nodes.Model import Model
from pam.pipelines.nodes.Output import Output


class WorkflowBase:

    # Initializer / Instance Attributes
    def __init__(self, site, model, metadata):
        self.site = site
        self.model = (model, metadata)

    def run(self):
        raise NotImplementedError


class Workflow(WorkflowBase):

    def run(self):
        experiment_dir = 'experiment_output/'
        output_dir = 'datasink'
        working_dir = 'workingdir'

        site = self.site
        model_path = self.model[0]
        model_metadata = str(self.model[1])

        # Infosource
        init_source = Node(IdentityInterface(fields=['site', 'model_path', 'model_metadata']), name="init_source")
        init_source.inputs.site = site
        init_source.inputs.model_path = os.path.abspath(model_path)
        init_source.inputs.model_metadata = model_metadata

        # Datasink - creates output folder for important outputs
        datasink = Node(DataSink(base_directory=experiment_dir, container=output_dir), name="datasink")

        wf = nipype.Workflow(name='workflow')
        wf.base_dir = opj(experiment_dir, working_dir)

        input = Node(Input(site=init_source.inputs.site), name="input")
        output = Node(Output(), name="output")
        model = Node(Model(model_path=init_source.inputs.model_path , model_metadata=init_source.inputs.model_metadata), name="model")

        # Connect all components of the workflow
        wf.connect([(init_source, input, [('site', 'site')]),
                    (input, model, [('folder_output', 'folder_input')]),
                    (init_source, model, [('model_path', 'model_path')]),
                    (init_source, model, [('model_metadata', 'model_metadata')]),
                    (model, output, [('out_put', 'in_file')])
                    ])
        wf.write_graph(graph2use='colored', format='png', simple_form=True)
        wf.run()
