Monoses
==============

This is an open source implementation of our unsupervised machine translation system, described in the following papers:
- Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2019. **[An Effective Approach to Unsupervised Machine Translation](https://www.aclweb.org/anthology/P19-1019.pdf)**. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 194-203.
- Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2018. **[Unsupervised Statistical Machine Translation](https://www.aclweb.org/anthology/D18-1399.pdf)**. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 3632-3642.

In addition, it also includes tools to induce bilingual lexica through unsupervised machine tranlation as described in the following paper:
- Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2019. **[Bilingual Lexicon Induction through Unsupervised Machine Translation](https://www.aclweb.org/anthology/P19-1494.pdf)**. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 5002-5007.

If you use this software for academic research, [please cite the relevant paper(s)](#publications).


Requirements
--------
- Python 3 with [PyTorch](https://pytorch.org/) (tested with v0.4) and [editdistance](https://github.com/roy-ht/editdistance), available from your `PATH`
- Java
- [Moses v4.0](http://www.statmt.org/moses/), compiled under `third-party/moses/`
- [FastAlign](https://github.com/clab/fast_align), compiled under `third-party/fast_align/build/`
- [Phrase2vec](https://github.com/artetxem/phrase2vec), compiled under `third-party/phrase2vec/`
- [VecMap](https://github.com/artetxem/vecmap), available under `third-party/vecmap/`
- [Fairseq](https://github.com/pytorch/fairseq) (tested with v0.6), available under `third-party/fairseq/`
- [Subword-NMT](https://github.com/rsennrich/subword-nmt), available under `third-party/subword-nmt/`
- [SacreBLEU](https://github.com/mjpost/sacrebleu), available under `third-party/sacrebleu/`.

A script is provided to download all the dependencies under `third-party/`:
```
./get-third-party.sh
```

Note, however, that the script only downloads their source code, and you will need to compile Moses (including [contrib/sigtest-filter](http://www.statmt.org/moses/?n=Advanced.RuleTables#ntoc5) and [moses2](http://www.statmt.org/moses/?n=Site.Moses2)), FastAlign and Phrase2vec yourself, and install Fairseq's dependencies. Please refer to the original documentation of each tool for detailed instructions on how to accomplish this.

In addition, you will also need to compile the tuning module in Java (which is based on [Z-MERT](http://cs.jhu.edu/~ozaidan/zmert/)) as follows:
```
cd training/tuning/zmert
make
```


Usage
--------

The following command trains an unsupervised machine translation system from monolingual corpora using the exact same settings described in our most recent paper:

```
python3 train.py --src SRC.MONO.TXT --src-lang SRC \
                 --trg TRG.MONO.TXT --trg-lang TRG \
                 --working MODEL-DIR
```

The parameters in the above command should be provided as follows:
- `SRC.MONO.TXT` and `TRG.MONO.TXT` are the source and target language monolingual corpora. You should just provide the raw text, and the training script will take care of all the necessary preprocessing (tokenization, deduplication etc.).
- `SRC` and `TRG` are the source and target language codes (e.g. 'en', 'fr', 'de'). These are used for language-specific corpus preprocessing using standard Moses tools.
- `MODEL-DIR` is the directory in which to save the output model.

By default, training uses 4 GPUs (with IDs 0, 1, 2 and 3) and takes about one week in our server. Once training is done, you can use the resulting model for translation as follows:

```
python3 translate.py MODEL-DIR --src SRC --trg TRG < INPUT.TXT > OUTPUT.TXT
```

In addition, you can also evaluate the model in the same settings as in our paper using the `evaluate.py` script.

For more details and additional options, run the above scripts with the `--help` flag.


### Bilingual Lexicon Induction

The following command induces a bilingual dictionary starting from a set of cross-lingual word embeddings using the exact same settings described in our paper:
```
python3 bli/induce-dictionary.py --embeddings SRC.EMB TRG.EMB \
                                 --corpus SRC.TOK.TXT TRG.TOK.TXT \
                                 --working OUTPUT-DIR
```

The parameters in the above command should be provided as follows:
- `SRC.EMB` and `TRG.EMB` are the input cross-lingual word embeddings. In our paper, these were obtained by training monolingual [fastText](https://fasttext.cc/) embeddings and mapping them using the unsupervised mode in [VecMap](https://github.com/artetxem/vecmap).
- `SRC.TOK.TXT` and `TRG.TOK.TXT` are the source and target language (monolingual) corpora used to train the embeddings above. You should provide the exact same preprocessed version used to train the embeddings.
- `OUTPUT-DIR` is the output directory in which to save the induced dictionaries (`src2trg.dic` and `trg2src.dic`) as well as the underlying machine translation model.


Publications
--------

If you use this software for academic research, please cite the relevant paper(s) as follows (in case of doubt, please cite `artetxe2019acl-umt`, and/or `artetxe2019acl-bli` if you use the bilingual lexicon induction code):
```
@inproceedings{artetxe2019acl-umt,
  author    = {Artetxe, Mikel  and  Labaka, Gorka  and  Agirre, Eneko},
  title     = {An Effective Approach to Unsupervised Machine Translation},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  month     = {July},
  year      = {2019},
  address   = {Florence, Italy},
  publisher = {Association for Computational Linguistics},
  pages     = {194--203},
  url       = {https://www.aclweb.org/anthology/P19-1019}
}

@inproceedings{artetxe2018emnlp,
  author    = {Artetxe, Mikel  and  Labaka, Gorka  and  Agirre, Eneko},
  title     = {Unsupervised Statistical Machine Translation},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  month     = {November},
  year      = {2018},
  address   = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics},
  pages     = {3632--3642},
  url       = {https://www.aclweb.org/anthology/D18-1399}
}

@inproceedings{artetxe2019acl-bli,
  author    = {Artetxe, Mikel  and  Labaka, Gorka  and  Agirre, Eneko},
  title     = {Bilingual Lexicon Induction through Unsupervised Machine Translation},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  month     = {July},
  year      = {2019},
  address   = {Florence, Italy},
  publisher = {Association for Computational Linguistics},
  pages     = {5002--5007},
  url       = {https://www.aclweb.org/anthology/P19-1494}
}
```

License
-------

Copyright (C) 2018-2020, Mikel Artetxe

Licensed under the terms of the GNU General Public License, either version 3 or (at your option) any later version. A full copy of the license can be found in LICENSE.txt.

The tuning module under `training/tuning/zmert/` is based on [Z-MERT](http://cs.jhu.edu/~ozaidan/zmert/).