# Availability, provenance and release boundaries

This repository is a curated record of original, author-supplied DAS research artifacts. It does **not** include or sublicense the original DAS recordings, original Phase-2 HDF5 shards or binary model checkpoints.

**Source data**: Tomasov et al., *Comprehensive Dataset for Event Classification Using Distributed Acoustic Sensing (DAS) Systems*, https://doi.org/10.1038/s41597-025-05088-4 . Consult the official dataset provider for current download and attribution conditions. Do not infer that a publication's license, dataset license and a separate third-party software license are necessarily identical.

**Research source code**: The authors or relevant rights holders must approve public distribution and select a LICENSE file before making a public code repository. No unapproved permissive or copyleft license has been created. In the absence of a code license, do not advertise the repository as freely reusable under an open-source license. This applies independently from permission to view publicly accessible research outputs.

**Privacy/security**: The curated notebooks have no stored execution outputs. Automatic static screening of this curated release is not a substitute for a rights-holder review of data, personal identifiers, credentials, and possible confidential comments before public publication.

**Reproducibility**: The included data support independent recomputation of 18 run-level stored event classification metrics and split/overlap checks; they do **not** demonstrate independently re-executed training or regeneration of the 16 omitted HDF5 shard bytes and stored checkpoint weights. The original source is supplied for reproducibility, with GPU execution and data assembly remaining prerequisites.

**Publication statements**: Include a public repository URL in the manuscript's Data Availability Statement only after that URL exists and is accessible to unconnected visitors. Do not insert an imaginary URL or claim the omitted shards/checkpoints are publicly hosted.

**Release gate:** This ZIP is a technically verified candidate for a public repository. Human author/rights-holder approval and assignment of software distribution terms remain separate steps and must occur before actual public upload. The GitHub workflow verifies only the saved outputs and manifest, not GPU training.
