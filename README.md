# Glyph

This repository is used to publish all the code used for the following article:

[Xiang Zhang, Yann LeCun, Which Encoding is the Best for Text Classification in Chinese, English, Japanese and Korean?, arXiv 1708.02657](https://arxiv.org/abs/1708.02657)

The code is not yet completely released. Will update here when it is done.

## Reproducibility Manifesto

If anyone sees a number in our paper, there is a script one can execute to reproduce it. No responsibility should be imposed on the user to figure out any experimental parameter barried in the paper's content.

## Datasets

The `data` directory contains the preprocessing scripts for all the datasets used in the paper. These datasets are released separately of their processing source code. See below for details.

### Summary

The following table is a summary of the datasets. Most of them have millions of samples for training.

| Dataset        | Language     | Classes | Train      | Test      |
|----------------|--------------|---------|------------|-----------|
| Dianping       | Chinese      | 2       | 2,000,000  | 500,000   |
| JD full        | Chinese      | 5       | 3,000,000  | 250,000   |
| JD binary      | Chinese      | 2       | 4,000,000  | 360,000   |
| Rakuten full   | Japanese     | 5       | 4,000,000  | 500,000   |
| Rakuten binary | Japanese     | 2       | 3,400,000  | 400,000   |
| 11st full      | Korean       | 5       | 750,000    | 100,000   |
| 11st binary    | Korean       | 2       | 4,000,000  | 400,000   |
| Amazon full    | English      | 5       | 3,000,000  | 650,000   |
| Amazon binary  | English      | 2       | 3,600,000  | 400,000   |
| Ifeng          | Chinese      | 5       | 800,000    | 50,000    |
| Chinanews      | Chinese      | 7       | 1,400,000  | 112,000   |
| NYTimes        | English      | 7       | 1,400,000  | 105,000   |
| Joint full     | Multilingual | 5       | 10,750,000 | 1,500,000 |
| Joint binary   | Multilingual | 2       | 15,000,000 | 1,560,000 |

### Download

Datasets are released separtely of the source code via links from Google Drive. *These datasets should only be used for the purpose of research*.

| Dataset        | Train                          | Test                          |
|----------------|--------------------------------|-------------------------------|
| Dianping       | [Link](https://goo.gl/uKPxyo)  | [Link](https://goo.gl/2QZpLx) |
| JD full        | [Link](https://goo.gl/u3vsak)  | [Link](https://goo.gl/hLZRky) |
| JD binary      | [Link](https://goo.gl/ZPj1ip)  | [Link](https://goo.gl/bqiEfP) |
| Rakuten full   | [Link](https://goo.gl/A7y14i)  | [Link](https://goo.gl/ve4mup) |
| Rakuten binary | [Link](https://goo.gl/3kYQ2f)  | [Link](https://goo.gl/m8FpeH) |
| 11st full      | [Link](https://goo.gl/F1oPBX)  | [Link](https://goo.gl/ZpTLND) |
| 11st binary    | [Link](https://goo.gl/8Qi7ao)  | [Link](https://goo.gl/nbBhFq) |
| Amazon full    | [Link](https://goo.gl/UzQWaj)  | [Link](https://goo.gl/EXkzWs) |
| Amazon binary  | [Link](https://goo.gl/u7AxWS)  | [Link](https://goo.gl/2fft8x) |
| Ifeng          | [Link](https://goo.gl/AtKsq4)  | [Link](https://goo.gl/tLWojy) |
| Chinanews      | [Link](https://goo.gl/1p4kdx)  | [Link](https://goo.gl/rxvhCJ) |
| NYTimes        | [Link](https://goo.gl/2hZeqd)  | [Link](https://goo.gl/66EDa5) |
| Joint full     | [Link](https://goo.gl/AJfzLC)  | [Link](https://goo.gl/mibMsV) |
| Joint binary   | [Link](https://goo.gl/YLMqNe)  | [Link](https://goo.gl/WRXQuJ) |

## GNU Unifont

The `glyphnet` scripts require the GNU Unifont character images to run. The file `unifont-8.0.01.t7b.xz` can be downloaded via [this link](https://goo.gl/aFxYHq).
