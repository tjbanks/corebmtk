from abc import abstractmethod
import os
import time
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)

from bmtk.simulator import bionet
from bmtk.simulator.bionet.io_tools import io
from bmtk.simulator.bionet import modules as mods
import bmtk.simulator.utils.simulation_reports as reports
import h5py
import neuron
from neuron import coreneuron
from neuron import h
from neuron.units import mV
import numpy as np

pc = h.ParallelContext()    # object to access MPI methods


class CoreMod():

    def __init__(self,*args,**kwargs):
        pass

    @abstractmethod
    def initialize(self,*args,**kwargs):
        pass

    @abstractmethod
    def finalize(self,*args,**kwargs):
        pass

class CoreSpikesMod(CoreMod):

    def __init__(self,*args,**kwargs):
        self.corenrn_all_spike_t = h.Vector()
        self.corenrn_all_spike_gids = h.Vector()
        self.spikes_file = kwargs.get('spikes_file')
        
    def initialize(self,*args,**kwargs):
        pc.spike_record(-1, self.corenrn_all_spike_t, self.corenrn_all_spike_gids )

    def finalize(self,*args,**kwargs):
        io.log_info('Saving spikes file to {}'.format(self.spikes_file))
        np_corenrn_all_spike_t = self.corenrn_all_spike_t.to_python()
        np_corenrn_all_spike_gids = self.corenrn_all_spike_gids.to_python()
        
        fp = h5py.File(self.spikes_file, "w")
        grp = fp.create_group('spikes/biophysical')
        grp.create_dataset('node_ids',data=list(np_corenrn_all_spike_gids))
        grp.create_dataset('timestamps',data=list(np_corenrn_all_spike_t))
        fp.close()

class CoreMembraneMod(CoreMod):

    def __init__(self,*args,**kwargs):
        self.variable_name = kwargs['variable_name']
        self.cells = kwargs['cells']
        self.sections = kwargs['sections']
        self.file_name = os.path.join(kwargs['tmp_dir'],kwargs['file_name'])

        self.time_vector = None
        self.record_dict = {}
        self.dt = 0
        self.tstop = 0

    def initialize(self,*args,**kwargs):
        self.time_vector = h.Vector().record(h._ref_t)
        
        for gid in self.cells:
            flag = pc.gid_exists(gid)
            if flag > 0 :
                cell = pc.gid2cell(gid)
                for variable in self.variable_name:
                    vec = h.Vector()
                    if variable == 'v':
                        vec.record(cell.soma[0](0.5)._ref_v) # TODO only allows for soma
                    else:
                        var_ref = getattr(cell.soma[0](0.5), variable)
                        vec.record(var_ref) # TODO only allows for soma

                    if not self.record_dict.get(variable):
                        self.record_dict[variable] = []

                    self.record_dict[variable].append(vec)

    def finalize(self,*args,**kwargs):
        io.log_info('Saving membrane report variables {} file to {}'.format(self.variable_name, self.file_name))
        fp = h5py.File(self.file_name, "w")

        mapping = fp.create_group('mapping')
        mapping.create_dataset('gids', self.cells)
        mapping.create_dataset('time', [0, self.tstop, self.dt])

        for variable, vec_list in self.record_dict.items():      
            data = []
            for vec in vec_list:
                np_vec = vec.to_python()
                data.append(list(np_vec))

            grp = fp.create_group(f"{variable}")
            grp.create_dataset('data',data=np.array(data).T)

        fp.close()

class CoreECPMod(mods.EcpMod):
    """
    Extracellular mechanisms are not allowed in CoreNEURON
    This module overwrites the existing framework by calculating
    the lfp at the end of the simulation by recording cell._ref_i_membrane_
    We skip the step phase and incorporate it into initialize and finalize
    """

    def __init__(self, *args,**kwargs):
        super(CoreECPMod, self).__init__(*args,**kwargs)
        self.cell_imvec = {} # gid:ecp
        self.file_name = kwargs['file_name']

    def initialize(self, sim):
        super(CoreECPMod, self).initialize(sim)
        # Start recording of im (usually retrieved from cell.get_im)
        for gid in self._cells:
            flag = pc.gid_exists(gid)
            if flag > 0 :
                cell = pc.gid2cell(gid)
                for sec in cell.all: # record _ref_i_membrane_ for each segment
                    for seg in sec:
                        vec = h.Vector()
                        vec.record(seg._ref_i_membrane_)
                
                        if not self.cell_imvec.get(gid):
                            self.cell_imvec[gid] = []
                        self.cell_imvec[gid].append(vec)

    def finalize(self, sim):
        io.log_info('Node saving ecp report to {}'.format(self.file_name))
        
        im_steps = {}

        for gid in self._local_gids:  # compute ecp only from the biophysical cells
            im_steps[gid] = np.array([vec for vec in self.cell_imvec[gid]]).T

        for n_time in range(sim.n_steps):

            if self._block_step == self._block_size:
                self.block(sim, (n_time-self._block_size, n_time))

            for gid in self._local_gids:

                im = im_steps[gid][n_time]
                tr = self._rel.get_transfer_resistance(gid)
                                
                # calculate the ecp/lfp in post processing since we now have a segment by segment recording
                ecp = np.dot(tr, im)

                if gid in self._saved_gids.keys():
                    # save individual contribution
                    self._saved_gids[gid][self._block_step, :] = ecp

                # add to total ecp contribution
                self._data_block[self._block_step, :] += ecp
            
            self._block_step +=1

        super(CoreECPMod, self).finalize(sim)



class CoreBioSimulator(bionet.BioSimulator):
    """
    A sub class implementation of bionet.BioSimulator compatible with CoreNeuron

    Use:
    Replace 
        sim = bionet.BioSimulator.from_config(conf, network=graph)
    With
        sim = CoreBioSimulator.from_config(conf, network=graph)
    """

    def __init__(self, network, dt, tstop, v_init, celsius, nsteps_block, start_from_state=False, gpu=False):
        super(CoreBioSimulator, self).__init__(network, dt, tstop, v_init, celsius, nsteps_block, start_from_state=False)

        io.log_info('Running core neuron sim')
        coreneuron.verbose = 3 # 3 equals debug mode

        self.config = None
        self.enable_core_mods = True
        self._core_mods = []  


    def __elapsed_time(self, time_s):
        if time_s < 120:
            return '{:.4} seconds'.format(time_s)
        elif time_s < 7200:
            mins, secs = divmod(time_s, 60)
            return '{} minutes, {:.4} seconds'.format(mins, secs)
        else:
            mins, secs = divmod(time_s, 60)
            hours, mins = divmod(mins, 60)
            return '{} hours, {} minutes and {:.4} seconds'.format(hours, mins, secs)

    def _init_mods(self):
        if not self.enable_core_mods:
            return

        for mod in self._core_mods:
           mod.initialize(sim=self)

    def _finalize_mods(self):
        if not self.enable_core_mods:
            return

        for mod in self._core_mods:
            mod.finalize(sim=self)


    def run(self):
        """
        Run the simulation
        """
        
        self.h.cvode.use_fast_imem(1)
        self._init_mods()

        self.start_time = h.startsw()
        s_time = time.time()
        pc.timeout(0)
         
        pc.barrier()  # wait for all hosts to get to this point
        io.log_info('Running simulation for {:.3f} ms with the time step {:.3f} ms'.format(self.tstop, self.dt))
        io.log_info('Starting timestep: {} at t_sim: {:.3f} ms'.format(self.tstep, h.t))
        io.log_info('Block save every {} steps'.format(self.nsteps_block))

        self.h.finitialize(self.v_init * mV)           
        pc.psolve(h.tstop)            
        pc.barrier()
        
        self._finalize_mods()
        pc.barrier()

        end_time = time.time()
        sim_time = self.__elapsed_time(end_time - s_time)
        io.log_info('Simulation completed in {} '.format(sim_time))

    @classmethod
    def from_config(cls, config, network, set_recordings=True, enable_core_mods=True, gpu=False):
        sim = super(CoreBioSimulator, cls).from_config(config, network, set_recordings=True)
        sim.enable_core_mods = enable_core_mods
        sim.config = config

        coreneuron.enable = True
        if gpu:
            coreneuron.gpu = True


        sim_reports = reports.from_config(config)
        for report in sim_reports:
            
            if isinstance(report, reports.SpikesReport):
                mod = CoreSpikesMod(**report.params)

            elif report.module == 'netcon_report':
                #mod = mods.NetconReport(**report.params)
                io.log_warning('Core Neuron BMTK Module {} not implemented, skipping.'.format(report.module))
                continue

            elif isinstance(report, reports.MembraneReport):
                if report.params['sections'] == 'soma':
                    mod = CoreMembraneMod(**report.params)
                    if report.params.get("cells") == 'all':
                        mod.cells = sim.biophysical_gids
                    mod.dt = sim.dt
                    mod.tstop = sim.tstop
                else:
                    #mod = mods.MembraneReport(**report.params)
                    io.log_warning('Core Neuron BMTK Module {} not implemented, skipping.'.format(report.module))
                    continue
            elif isinstance(report, reports.ClampReport):
                #mod = mods.ClampReport(**report.params)
                io.log_warning('Core Neuron BMTK Module {} not implemented, skipping.'.format(report.module))
                continue

            elif isinstance(report, reports.ECPReport) or report.module == 'ecp':
                #mod = mods.EcpMod(**report.params)
                mod = CoreECPMod(**report.params)
                if report.params.get("cells") == 'all':
                    mod._cells = list(sim.biophysical_gids)

                def setup_ecp(cell):
                    # Same as cell.setup_ecp BUT does not inserte extracellular mech
                    cell.im_ptr = h.PtrVector(cell.morphology.nseg)  # pointer vector
                    # used for gathering an array of  i_membrane values from the pointer vector
                    cell.im_ptr.ptr_update_callback(cell.set_im_ptr)
                    cell.imVec = h.Vector(cell.morphology.nseg)

                # Set up the ability for ecp on all relevant cells
                for gid, cell in network.cell_type_maps('biophysical').items():
                    setup_ecp(cell)
                    

            elif report.module == 'save_synapses':
                #mod = mods.SaveSynapses(**report.params)
                io.log_warning('Core Neuron BMTK Module {} not implemented, skipping.'.format(report.module))
                continue

            else:
                # TODO: Allow users to register customized modules using pymodules
                io.log_warning('Unrecognized module {}, skipping.'.format(report.module))
                continue

            sim._core_mods.append(mod)
            
        return sim