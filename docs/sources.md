# Sources, licenses and actual tool use

## Dataset attribution

PneumoGuard AI uses the **PneumoniaMNIST** subset only, at 28×28 grayscale resolution. It preserves the official labels and splits. Displayed examples are MedMNIST's transformed source images enlarged for readability; enlarging adds no image information.

1. **MedMNIST data and metadata:** [official repository](https://github.com/MedMNIST/MedMNIST), [metadata](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py), [official Zenodo distribution](https://zenodo.org/records/10519652). PneumoniaMNIST data license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). MedMNIST Python library code license: Apache 2.0. These are distinct licenses for distinct materials.
2. **MedMNIST v2:** Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister and Bingbing Ni (2023). *MedMNIST v2—A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data 10, 41. [DOI](https://doi.org/10.1038/s41597-022-01721-8).
3. **MedMNIST v1:** Jiancheng Yang, Rui Shi and Bingbing Ni (2021). *MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis.* IEEE ISBI, pp. 191–195. [DOI](https://doi.org/10.1109/ISBI48211.2021.9434062). The MedMNIST authors request citation of both v1 and v2 and the relevant source paper.
4. **Underlying data:** Daniel Kermany, Kang Zhang and Michael Goldbaum (2018). *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*, Mendeley Data, version 2. [DOI](https://doi.org/10.17632/rscbjbr9sj.2); [source record](https://data.mendeley.com/datasets/rscbjbr9sj/2). The source record lists CC BY 4.0. Only the chest X-ray derivatives in PneumoniaMNIST are used, not the OCT images.
5. **Source research:** Kermany et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell 172(5), 1122–1131.e9. [DOI](https://doi.org/10.1016/j.cell.2018.02.010).

The dataset is not intended for clinical use. Source annotations are used as benchmark references. Patient-level separation cannot be independently verified from the supplied arrays. No private patient data is collected. Do not commit the downloaded dataset; retain attribution with exported example figures and cite these sources when sharing them.

## Libraries and platforms

The implementation uses Python and its standard library, PyTorch/torchvision, medmnist, NumPy, pandas, scikit-learn, Matplotlib, Pillow and ipywidgets. Notebook structure and execution checks use nbformat, nbclient, IPython/ipykernel and pytest. Google Colab is the intended student runtime; GitHub is an optional manually managed submission destination. Exact experiment versions are generated in `results/environment.json`, not guessed in advance. Version ranges are specified in `requirements.txt`.

Primary documentation: [Python](https://docs.python.org/3/), [PyTorch](https://pytorch.org/docs/stable/), [torchvision](https://pytorch.org/vision/stable/), [MedMNIST](https://github.com/MedMNIST/MedMNIST), [NumPy](https://numpy.org/doc/), [pandas](https://pandas.pydata.org/docs/), [scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html), [Matplotlib](https://matplotlib.org/stable/), [Pillow](https://pillow.readthedocs.io/), [ipywidgets](https://ipywidgets.readthedocs.io/), [nbformat](https://nbformat.readthedocs.io/), [nbclient](https://nbclient.readthedocs.io/), [pytest](https://docs.pytest.org/).

No tutorial implementation, pretrained weights, paid model API or existing project code was reused. Selective classification is an established idea; the project's contribution is this original experiment, validation policy, workload evaluation and benchmark explorer, not an invention claim about abstention itself.

## AI and development assistance

**ChatGPT and OpenAI Codex** were used to create original code, design the experiment, write and review tests, and prepare documentation. Parallel Codex agents reviewed scientific logic and prepared tests and documentation. This is assistance, not an external clinical collaboration. The student should review the implementation, reproduce the results and disclose the tools according to competition requirements.

Development also used a managed Codex Cloud Python environment, shell tools, package downloads and a web retrieval tool to read primary-source competition and dataset pages. No patient, hospital, doctor, sponsor, team membership, endorsement or award is implied.

## Competition reference

The [official briefs and submission HTML page](https://highschoolhackathon.org/briefs) was accessible during this session. The linked Google Drive project-brief PDF showed a loading/sign-in surface and could not be read; the supplied detailed task brief was used as the implementation specification. The student must read the complete official brief and check current requirements before submitting.

This project is a prototype and is not medical advice.
