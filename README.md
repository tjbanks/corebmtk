# corebmtk
A module to allow BMTK to function with Core Neuron before official support.

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