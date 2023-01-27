# corebmtk
A module to allow BMTK to function with Core Neuron before official support.

Currently, BMTK uses `h.run` and `extracellular` NEURON mechanisms, which are both incompatible with CoreNeuron.

`corebmtk` uses `pc.psolve` and takes an alternate approach to recording and storing output. The goal of this project
was to maintain all of BMTK's functionality while allowing for the use of CoreNeuron.

### Installation

```
pip install --upgrade corebmtk
```

### Usage

In your `run_network.py` `BMTK` script replace your BioSimulator with a CoreBioSimulator.

```
import corebmtk

# sim = bionet.BioSimulator.from_config(conf, network=graph)
sim = corebmtk.CoreBioSimulator.from_config(conf, network=graph)
```

### Limitations

Some recoring mechanisms are not yet implemented. See run output for more info.

Implemented recording modules:

1. `reports.SpikesReport`
1. `netcon_report`
1. `reports.MembraneReport (report.params['sections'] == 'soma')`
1. `reports.ECPReport`

Modules NOT implemented

1. `mods.MembraneReport` (general)
1. `reports.ClampReport`
1. `save_synapses`

### Troubleshooting

When using ECP recording: 
```CoreNEURON cannot simulate a model that contains the extracellular mechanism```

Change your `ecp report` to use `"module": "ecp"` instead of `"module": "extracellular"`

```
    "ecp": {
      "cells": "all",
      "variable_name": "v",
      "module": "ecp",
      "electrode_positions": "./components/recXelectrodes/linear_electrode.csv",
      "file_name": "ecp.h5",
      "electrode_channels": "all"
    }
```

### Acknowledgments

Thanks [@gregglickert](https://github.com/gregglickert) for kicking this project off.

[BMTK](https://github.com/AllenInstitute/bmtk/)