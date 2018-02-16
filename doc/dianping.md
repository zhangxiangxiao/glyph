# Dianping

This documentation contains information on how to reproduce all the results for the `Dianping` datasets in the paper.

The root directory `/` in this documentation indicates the root directory of this repository.

## Download the dataset

Original text data for training and testing are available via these two links: [`train.csv.xz`](https://goo.gl/uKPxyo) [`test.csv.xz`](https://goo.gl/2QZpLx). When you download them, make sure to put them in the `/data/data/dianping` directory and unxz so that you have `train.csv` and `test.csv` available.

In the case that some preprocessing scripts may not output exactly the same results due to intrinsic randomness, we also provide the preprocessed intermediate results for your convenience in the documentation. These intrinsic randomness may include inconsistencies between versions of toolkits or random orders when executing multi-threaded processing scripts.

## GlyphNet

This section introduces how to prepare and run GlyphNet experiments.

### Prepare GNU Unifont

Running the glyphnet training script requires the GNU Unifont character images. We have built these images into a Torch 7 binary serialization file and it can be download via this link: [`unifont-8.0.1.t7b.xz`](https://goo.gl/aFxYHq). After downloading, put it in `/unifont/unifont` directory and unxz so that you have `unifont-8.0.1.t7b` available.

### Build Byte Serialization Files

The next step is to build the serialized code files. The first step is to build the string serialization files. Switch to the `/data/dianping` directory, then execute the following commands

```bash
th construct_string.lua ../data/dianping/train.csv ../data/dianping/train_string.t7b
th construct_string.lua ../data/dianping/test.csv ../data/dianping/test_string.t7b
```

These 2 commands will build byte serialization files for the samples in its original language. It assumes the texts are contained in a comma-separated-value format in which the first field is treated as the class index (starting from 1), and the remaining fields are all texts.

The output files contain a lua table that has the following members

* `index`: a table that contains index tensors for each class. For example `index[i]` is an n x m x 2 `LongTensor` that contains the starting position and length of byte string representing each sample in class i. We assume that class i contains n samples, and there are m text fields in the CSV file.
* `content`: a `ByteTensor` that contains the serialization of the strings of all samples. Each string is ended with a 0 byte, which is not included in the length count in `index`.

### Build Unicode Serialization Files

From this byte-level serialization, we will be able to construct serialization files that contain unicode values to be used in the `glyphnet` training scripts. To do this, execute the following 2 commands

```bash
th construct_code.lua ../data/dianping/train_string.t7b ../data/dianping/train_code.t7b
th construct_code.lua ../data/dianping/test_string.t7b ../data/dianping/test_code.t7b
```

Each of these code files contain a lua table that has 2 `LongTensor` members: `code` and `code_value`. The have a similar structure as the `index` and `content` members of the byte serialization files, but in this case they are for unicode values.

### Execute the Experiments

Then, you can switch to `/glyphnet`, and execute the following scripts to run the training program for the large GlyphNet

```bash
mkdir -p models/dianping/spatial8temporal12length512feature256
./archive/dianping_spatial8temporal12length512feature256.sh
```

The first command simply creates a directory where checkpointing files will be written into during training. Note that the shell scripts also accepts command-line parameters and can pass it directly to the training program. The most useful ones are probably `-driver_visualize false` and `-driver_plot false`, that disable visualization and plotting so that you can run the training programs on a headless server.

Similarly, the following commands execute the experiment for the small GlyphNet

```bash
mkdir -p models/dianping/spatial6temporal8length486feature256
./archive/dianping_spatial6temporal8length486feature256.sh
```

## OnehotNet

This section details how to execute OnehotNet experiments. Note that OnehotNet in this article are operating at byte-level for either the original text or the romanized text. In the case of romanized text, it is the same as character-level.

### Byte-Level OnehotNet for Original Text

To train OnehotNet for the original text, we only need the previously built byte serialization files. If you do not have them, see previous sections for using `construct_string.lua` data processing scripts.

#### Execute the Experiments

Assuming your current working directory is `/onehotnet`, the following commands execute experiments for large OnehotNet on the original text samples.

```bash
mkdir -p models/dianping/onehot4temporal12length2048feature256
./archive/dianping_onehot4temporal12length2048feature256.sh
```

Similarly, the small OnehotNet experiments can be done using the following commands

```bash
mkdir -p models/dianping/onehot4temporal8length1944feature256
./archive/dianping/onehot4temporal8length19444feature256.sh
```

### Character-Level OnehotNet for Romanized Text

This section details how to execute OnehotNet for romanized text. But before that, we need to build the romanized data first.

#### Build Romanized Text Serialization Files

The first step is to convert the original text into a romanization format. This is done in this project automatically using the [`pypinyin`](https://github.com/mozillazg/python-pinyin) package (version 0.12 for the results in the paper). You also want to install [`jieba`](https://github.com/fxsjy/jieba) (version 0.38 for the results in the paper) so that `pypinyin` can use it for word segmentation. All these packages were installed in a Python 3 environment.

Switch the working directory to `/data/dianping`, the following commands converting the original text to a romanization format for the Dianping dataset.

```bash
python3 construct_pinyin.py -i ../data/dianping/train.csv -o ../data/dianping/train_pinyin.csv
python3 construct_pinyin.py -i ../data/dianping/test.csv -o ../data/dianping/test_pinyin.csv
```

For you convenience, we also release prebuilt files [`train_pinyin.csv.xz`](https://goo.gl/zZDfXq) and [`test_pinyin.csv.xz`](https://goo.gl/JjKEht) in case the previous pipelines do not produce exactly the same romanization as they did for the results in the paper.

Then, we can use `construct_string.lua` again for constructing the byte serialization of romanized texts.

```bash
th construct_string.lua ../data/dianping/train_pinyin.csv ../data/dianping/train_pinyin_string.t7b
th construct_string.lua ../data/dianping/test_pinyin.csv ../data/dianping/test_pinyin_string.t7b
```

#### Execute the Experiments

Assuming your current working directory is `/onehotnet`, the following commands execute experiments for large OnehotNet on the romanized text samples.

```bash
mkdir -p models/dianping/onehot4temporal12length2048feature256roman
./archive/dianping_onehot4temporal12length2048feature256roman.sh
```

Similarly, the small OnehotNet experiments can be done using the following commands

```bash
mkdir -p models/dianping/onehot4temporal8length1944feature256roman
./archive/dianping/onehot4temporal8length19444feature256roman.sh
```

## EmbedNet

This section introduces how to build the data files and executing experiments for EmbedNet.

### Character-Level EmbedNet for Original Text

Since we already built the serialization data files for unicode characters for GlyphNet, we can directly use them. The only step required is to run the commands for training the models.

Assuming the current working directory is `/embednet`, the following commands will start the training process for large character-level EmbedNet.

```bash
mkdir -p models/dianping/temporal12length512feature256
./archive/dianping_temporal12length512feature256.sh
```

And for small character-level EmbedNet

```bash
mkdir -p models/dianping/temporal8length486feature256
./archive/dianping_temporal8length486feature256.sh
```

### Byte-Level EmbedNet for Original Text

This section details how to train byte-level EmbedNet for the original text

#### Convert Byte Serialization Files

Since the EmbedNet training program assumes the data files contain a table of 2 members `code` and `code_value`, we need to change the variable names in the string serialization files to match this. This can be done in `/data/dianping` by executing the following commands

```bash
th convert_string_code.lua ../data/dianping/train_string.t7b ../data/dianping/train_string_code.t7b
th convert_string_code.lua ../data/dianping/test_string.t7b ../data/dianping/test_string_code.t7b
```

#### Execute the Experiments

Assuming the current working director is `/embednet`, the following commands start the training process for the large byte-level EmbedNet

```bash
mkdir -p models/dianping/temporal12length512feature256byte
./archive/dianping_temporal12length512feature256byte.sh
```

And for small byte-level EmbedNet

```bash
mkdir -p models/dianping/temporal8length486feature256byte
./archive/dianping_temporal8length486feature256byte.sh
```

### Character-Level EmbedNet for Romanized Text

Note that characters for romanized text is the same as bytes. Therefore, the steps are exactly the same as the byte-level EmbedNet, except for romanized text instead of original text.

#### Convert Byte Serialization Files

In `/data/dianping`, execute the following commands

```bash
th convert_string_code.lua ../data/dianping/train_pinyin_string.t7b ../data/dianping/train_pinyin_string_code.t7b
th convert_string_code.lua ../data/dianping/test_pinyin_string.t7b ../data/dianping/test_pinyin_string_code.t7b
```

#### Execute the Experiments

Assuming the current working director is `/embednet`, the following commands start the training process for the large character-level EmbedNet for romanized text

```bash
mkdir -p models/dianping/temporal12length512feature256roman
./archive/dianping_temporal12length512feature256roman.sh
```

And for small EmbedNet

```bash
mkdir -p models/dianping/temporal8length486feature256roman
./archive/dianping_temporal8length486feature256roman.sh
```

### Word-Level Embednet for Original Text

This section introduces how to segment word from the text, build the word serialization files, and execute the commands.

#### Build Word Serialization Files for Original Text

The first step for building the word serialization files is to segment the words. This is done by executing a Python 3 script as follows, assuming you have the [`jieba`](https://github.com/fxsjy/jieba) package installed (version 0.38 for the results in the paper) and the working directory is `/data/dianping`.

```bash
python3 segment_word.py -i ../data/dianping/train.csv -o ../data/dianping/train_word.csv -l ../data/dianping/train_word_list.csv
python3 segment_word.py -i ../data/dianping/test.csv -o ../data/dianping/test_word.csv -l ../data/dianping/train_word_list.csv -r
```

The first command generate 2 data files. `train_word.csv` is a file containing sequences of indices of segmented words from the original text fields, whereas `train_word_list.csv` contains the list of words. The second command read the same list of words generated from the training data (therefore the `-r` option) and use that list to build sequences for the testing data. This is done deliberately so that new words not in the training data are not considered for classification results.

In case you do not obtain the same word segmentation results as in the paper, we also release files [train_word.csv.xz](https://goo.gl/7nD3aJ), [test_word.csv.xz](https://goo.gl/49GB24) and [train_word_list.csv.xz](https://goo.gl/g4JF1B) here.

The second step is to build the word serialization files from the segmentation results.

```bash
th construct_word.lua ../data/dianping/train_word.csv ../data/dianping/train_word.t7b
th construct_word.lua ../data/dianping/test_word.csv ../data/dianping/test_word.t7b
```

#### Execute the Experiments

When we have `train_word.t7b` and `test_word.t7b`, we can start executing the experiments for word-level EmbedNet models. Assume that the current directory is `/embednet`, the following commands start the training process for the large word-level EmbedNet for original text

```bash
mkdir -p models/dianping/temporal12length512feature256word
./archive/dianping_temporal12length512feature256word.sh
```

And for small EmbedNet

```bash
mkdir -p models/dianping/temporal8length486feature256word
./archive/dianping_temporal8length486feature256word.sh
```

### Word-Level EmbedNet for Romanized Text

Similar to the original text, romanized text also require word segmentation before being able to pass through the EmbedNet training program.

#### Build Word Serialization Files for Romanized Text

Word segmentation for romanized text is pretty simple. Assume you are in `/data/dianping`, the following commands do the job

```bash
th segment_roman_word.lua ../data/dianping/train_pinyin.csv ../data/dianping/train_pinyin_word.csv ../data/dianping/train_pinyin_word_list.csv
th segment_roman_word.lua ../data/dianping/test_pinyin.csv ../data/dianping/test_pinyin_word.csv ../data/dianping/train_pinyin_word_list.csv true
```

Note the additional `true` argument in the second command-line to inform the script to use the training word list for constructing the indices for the testing data.

Then, word serialization files can be built from the segmentation results using the following commands.

```bash
th construct_word.lua ../data/dianping/train_pinyin_word.csv ../data/dianping/train_pinyin_word.t7b
th construct_word.lua ../data/dianping/test_pinyin_word.csv ../data/dianping/test_pinyin_word.t7b
```

#### Execute the Experiments

When we have `train_pinyinword.t7b` and `test_pinyinword.t7b`, we can start executing the experiments for word-level EmbedNet models. Assume that the current directory is `/embednet`, the following commands start the training process for the large word-level EmbedNet for original text

```bash
mkdir -p models/dianping/temporal12length512feature256romanword
./archive/dianping_temporal12length512feature256romanword.sh
```

And for small EmbedNet

```bash
mkdir -p models/dianping/temporal8length486feature256romanword
./archive/dianping_temporal8length486feature256romanword.sh
```

## Linear Model

### Character-Level 1-Gram Linear Model for Original Text

### Character-Level 5-Gram Linear Model for Original Text

### Word-Level 1-Gram Linear Model for Original Text

### Word-Level 5-Gram Linear Model for Original Text

### Word-Level 1-Gram Linear Model for Romanized Text

### Word-Level 5-Gram Linear Model for Romanized Text

## fastText

### Character-Level fastText for Original Text

### Word-Level fastText for Original Text

### Word-Level fastText for Romanized Text
