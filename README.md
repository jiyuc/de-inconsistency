# De-inconsistency for GOA

Automatic detection of inconsistency for literature-based Gene Onology Annotation [#paperwithcode]()







## Intoduction

This demo illustrates the detection of self-consistency and four types of semantic inconsistency between a triplet of GO term, gene product, and direct literature evidence in GOA instance.

For example, giving an inconsistent GOA as below:

🧬 Mtor (GeneID:36264) \
📖 At anaphase, 𝑀𝑡𝑜𝑟 plays a role in spindle elongation, thereby affecting normal chromosome movement. (PMID: 	19273613) \
🏷️ spindle (GO:0005819)

🤔Decision: irrelevant GO mention

Reason: the evidence does not indicate 𝑀𝑡𝑜𝑟 locates at ``spindle''.

## Method
![header](images/header.png)







## Authors

- [Jiyu Chen](https://jiyuc.live)
- Benjamin Goudey
- Nicholas Geard
- Justin Zobel
- Karin Verspoor



## Citation

Please consider citing one of the published work:

- Chen, J., Goudey, B., Zobel, J., Geard, N. and Verspoor, K., 2023. Integrating Background Knowledge for detection of inconsistency for literature-based Gene Ontology Annotation, Manuscript submitted.

- [Chen, J., Goudey, B., Zobel, J., Geard, N. and Verspoor, K., 2022. Exploring automatic inconsistency detection for literature-based gene ontology annotation. Bioinformatics, 38(Supplement_1), pp.i273-i281.](https://academic.oup.com/bioinformatics/article/38/Supplement_1/i273/6617491)

- [Chen, J., Geard, N., Zobel, J. and Verspoor, K., 2021. Automatic consistency assurance for literature-based gene ontology annotation. BMC bioinformatics, 22(1), pp.1-22](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04479-9)





