# Learning Geometry-Aware Representations for New Intent Discovery
This is the implementation of our ACL 2024 paper GeoID.

### Requirements

After creating a virtual environment, run

```
pip install -r requirements.txt
```

### Data 

refer to [fanolabs/NID_ACLARR2022](https://github.com/fanolabs/NID_ACLARR2022?tab=readme-ov-file)

### Pretrain

You can download the pretrained checkpoints from following https://drive.google.com/file/d/1dLiQPDFcP_TSnEemhjDzPYaqMr6sSWW2/view?usp=drive_link. And then put them into a folder `pretrained_models` in root directory. 

### How to run

BANKING dataset as an example 

```
sh scripts/run_banking.sh
```



### How to cite 

```
@inproceedings{tang-etal-2024-learning,
    title = "Learning Geometry-Aware Representations for New Intent Discovery",
    author = "Tang, Kai  and
      Zhao, Junbo  and
      Ding, Xiao  and
      Wu, Runze  and
      Feng, Lei  and
      Chen, Gang  and
      Wang, Haobo",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.306",
    doi = "10.18653/v1/2024.acl-long.306",
    pages = "5641--5654"
}
```

