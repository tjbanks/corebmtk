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

class CoreSpikesMod(mods.SpikesMod):
    """
    Class continues to function as is, just need to run block before output.
    """
    def __init__(self, *args,**kwargs):
        super(CoreSpikesMod, self).__init__(*args,**kwargs)
        
    def initialize(self,*args,**kwargs):
        super(CoreSpikesMod, self).initialize(*args,**kwargs)

    def finalize(self,*args,**kwargs):
        self.block(kwargs['sim'],None)
        super(CoreSpikesMod, self).finalize(*args,**kwargs)


class CoreNetconReport(mods.NetconReport):

    def __init__(self, *args,**kwargs):
        super(CoreNetconReport, self).__init__(*args,**kwargs)
        self.record_dict = {} # {gid:{varname:[record list]}}
        
    def initialize(self,*args,**kwargs):
        super(CoreNetconReport, self).initialize(*args,**kwargs)

        for gid, netcon_objs in self._object_lookup.items(): # _object_lookup? or local_cells
            for var_name in self._variables:
                
                vecs = []
                for syn in netcon_objs:
                    vec = h.Vector()
                    var_ref = getattr(syn, f'_ref_{var_name}')
                    vec.record(var_ref)
                    vecs.append(vec)

                if not self.record_dict.get(gid):
                    self.record_dict[gid] = {}

                self.record_dict[gid][var_name] = vecs


    def finalize(self,*args,**kwargs):
        for gid, netcon_objs in self._object_lookup.items():
            for var_name in self._variables:
                self.record_dict[gid][var_name] = np.array(self.record_dict[gid][var_name]).T
        
        sim = kwargs['sim']
        for tstep in range(sim.n_steps):
            
            # save all necessary cells/variables at the current time-step into memory
            if not self._record_on_step(tstep):
                return

            for gid, netcon_objs in self._object_lookup.items():
                pop_id = self._gid_map.get_pool_id(gid)
                for var_name in self._variables:
                    #syn_values = [getattr(syn, var_name) for syn in netcon_objs]
                    syn_values = list(self.record_dict[gid][var_name][tstep])
                if syn_values:
                    self._var_recorder.record_cell(
                        pop_id.node_id,
                        population=pop_id.population,
                        vals=syn_values,
                        tstep=self._curr_step
                    )

            self._curr_step += 1

        self.block(kwargs['sim'],None)
        super(CoreNetconReport, self).finalize(*args,**kwargs)


class CoreSomaReport(mods.SomaReport):

    def __init__(self, *args,**kwargs):
        super(CoreSomaReport, self).__init__(*args,**kwargs)
        self.record_dict = {} # gid:{variable:vector}
        self.h.cvode.cache_efficient(1)

    def initialize(self, sim):
        super(CoreSomaReport, self).initialize(sim)

        for gid in self._local_gids:
            flag = pc.gid_exists(gid)
            if flag > 0 :
                cell = pc.gid2cell(gid)
                for variable in self._variables:
                    vec = h.Vector()
                    if variable == 'v':
                        vec.record(cell.soma[0](0.5)._ref_v)
                    else:
                        var_ref = getattr(cell.soma[0](0.5), variable)
                        vec.record(var_ref)

                    if not self.record_dict.get(gid):
                        self.record_dict[gid] = {}

                    self.record_dict[gid][variable] = vec

    def finalize(self, sim):
        io.log_info('Node saving CoreSomaReport to {}'.format(self._file_name))
        
        # reformat self.record_dict to be a bit easier to deal with
        for gid, variables in self.record_dict.items():
            for var, vector in variables.items():
                self.record_dict[gid][var] = list(np.array(vector))

        for tstep in range(sim.n_steps):
            
            # save all necessary cells/variables at the current time-step into memory
            if not self._record_on_step(tstep):
                return

            for gid in self._local_gids:
                pop_id = self._gid_map.get_pool_id(gid)
                cell = sim.net.get_cell_gid(gid)
                for var_name in self._variables:
                    #var_val = getattr(cell.hobj.soma[0](0.5), var_name)
                    var_val = self.record_dict[gid][var_name][tstep]
                    self._var_recorder.record_cell(
                        pop_id.node_id,
                        population=pop_id.population,
                        vals=[var_val],
                        tstep=self._curr_step
                    )

                for var_name, fnc in self._transforms.items():
                    #var_val = getattr(cell.hobj.soma[0](0.5), var_name)
                    var_val = self.record_dict[gid][var_name][tstep]
                    new_val = fnc(var_val)
                    self._var_recorder.record_cell(
                        pop_id.node_id,
                        population=pop_id.population,
                        vals=[new_val],
                        tstep=self._curr_step)
        self.block(sim,None)
        super(CoreSomaReport, self).finalize(sim)

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
        for gid in self._local_gids:
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
        
        ecp_steps = {}

        for gid in self._local_gids:  # compute ecp only from the biophysical cells
            tr = self._rel.get_transfer_resistance(gid)
            ecp_steps[gid] = np.tensordot(np.array([vec for vec in self.cell_imvec[gid]]).T,tr,axes=((1,1)))

        for n_time in range(sim.n_steps):

            if self._block_step == self._block_size:
                self.block(sim, (n_time-self._block_size, n_time))

            for gid in self._local_gids:

                ecp = ecp_steps[gid][n_time][0]

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
    import corebmtk

    Replace 
        sim = bionet.BioSimulator.from_config(conf, network=graph)
    With
        sim = corebmtk.CoreBioSimulator.from_config(conf, network=graph)
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
        sim = super(CoreBioSimulator, cls).from_config(config, network, set_recordings=set_recordings)
        sim.enable_core_mods = enable_core_mods
        sim.config = config

        coreneuron.enable = True # We assume we're using core neuron if you're using the CoreBioSimulator class
        if gpu:
            coreneuron.gpu = True


        sim_reports = reports.from_config(config)
        for report in sim_reports:
            
            if isinstance(report, reports.SpikesReport):
                mod = CoreSpikesMod(**report.params)

            elif report.module == 'netcon_report':
                #mod = mods.NetconReport(**report.params)
                mod = CoreNetconReport(**report.params)

            elif isinstance(report, reports.MembraneReport):
                if report.params['sections'] == 'soma':
                    mod = CoreSomaReport(**report.params)
                    
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