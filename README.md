Monoses
==============

This is an open source implementation of our unsupervised statistical machine translation system, described in the following paper:

Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2018. **[Unsupervised Statistical Machine Translation](https://arxiv.org/pdf/1809.01272.pdf)**. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP 2018)*.

If you use this software for academic research, please cite the paper in question:
```
@inproceedings{artetxe2018emnlp,
  author    = {Artetxe, Mikel  and  Labaka, Gorka  and  Agirre, Eneko},
  title     = {Unsupervised Statistical Machine Translation},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  month     = {November},
  year      = {2018},
  address   = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics}
}
```


Requirements
--------
- Python 3 with [PyTorch](https://pytorch.org/) (tested with v0.4), available from your `PATH`
- [Moses v4.0](http://www.statmt.org/moses/), compiled under `third-party/moses/`
- [FastAlign](https://github.com/clab/fast_align), compiled under `third-party/fast_align/build/`
- [Phrase2vec](https://github.com/artetxem/phrase2vec), compiled under `third-party/phrase2vec/`
- [VecMap](https://github.com/artetxem/vecmap), available under `third-party/vecmap/`

A script is provided to download all the dependencies under `third-party/`:
```
./get-third-party.sh
```

Note, however, that the script only downloads their source code, which you still need to compile yourself. Please refer to the original documentation of each tool for detailed instructions on how to accomplish this. 


Usage
--------

The following command trains an unsupervised SMT system from monolingual corpora using the exact same settings described in the paper:

```
python3 train.py --src SRC.MONO.TXT --src-lang SRC \
                 --trg TRG.MONO.TXT --trg-lang TRG \
                 --working MODEL-DIR
```

The parameters in the above command should be provided as follows:
- `SRC.MONO.TXT` and `TRG.MONO.TXT` are the source and target language monolingual corpora. You should just provide the raw text, and the training script will take care of all the necessary preprocessing (tokenization, deduplication etc.).
- `SRC` and `TRG` are the source and target language codes (e.g. 'en', 'fr', 'de'). These are used for language-specific corpus preprocessing using standard Moses tools.
- `MODEL-DIR` is the directory in which to save the output model.

Using the above settings, training takes about one week in our modest server. Once training is done, you can use the resulting model for translation as follows:

```
python3 translate.py MODEL-DIR --src SRC --trg TRG < INPUT.TXT > OUTPUT.TXT
```

For more details and additional options, run the above scripts with the `--help` flag.


License
-------

Copyright (C) 2018, Mikel Artetxe

Licensed under the terms of the GNU General Public License, either version 3 or (at your option) any later version. A full copy of the license can be found in LICENSE.txt.
