# Quality Proxies Framework
[Data](/Data) and [Code](/Code) for `"Garbage, Glitter, or Gold: Assigning Multi-dimensional QualityScores to Social Media Seeds for Web Archive Collections"` paper.
## Evaluation Dataset
The following Table replicates Table 7 and includes links to the dataset topics for Reference (gold standard) and Twitter seeds.
| Is Gold Standard |     Source     |        Topic       |    Extraction-Range   |     Seed Count    |
|:----------------:|:--------------:|:------------------:|:---------------------:|:-----------------:|
|        Yes       |     Google     |  [hurricane harvey](/Dataset/google/all/20200412025044_hurricane_harvey/hurricane_harvey_qp_mc.json.ss)  |       2020-04-11      | 199 (Page 1 - 20) |
|        Yes       |     Google     | [flint water crisis](/Dataset/google/all/20200410034257_flint_water_crisis/flint_water_crisis_qp_mc.json.ss) |       2020-04-10      | 173 (Page 1 - 20) |
|        Yes       |     Google     |     [coronavirus](/Dataset/google/all/20200410154858_coronavirus/coronavirus_qp_mc.json.ss)    |       2020-04-09      | 176 (Page 1 - 20) |
|        Yes       |     Google     |   [2018 world cup](/Dataset/google/all/20190109000000_2018_world_cup/2018_world_cup_qp_mc.json.ss)   |       2019-01-09      | 112 (Page 1 - 10) |
|        Yes       |     Google     |     [ebola virus](/Dataset/google/all/20171231000000_ebola_virus/ebola_virus_qp_mc.json.ss)    |       2017-11-29      |  97 (Page 1 - 10) |
|        Yes       |     Google     |  [hurricane harvey](/Dataset/google/all/20170901000000_hurricane_harvey/hurricane_harvey_qp_mc.json.ss)  |   2017-09-(02 to 29)  |    51 (Page 1)    |
|        Yes       |   Archive-It   |     [coronavirus](/Dataset/human/archiveit/20200410154858_coronavirus/coronavirus_qp_mc.json.ss)    |       2020-03-15      |        574        |
|        Yes       |   Archive-It   |  [hurricane harvey](/Dataset/human/archiveit/20170901000000_hurricane_harvey/hurricane_harvey_qp_mc.json.ss)  | 2017-(08-25 to 09-29) |         37        |
|        Yes       |   Archive-It   |     [ebola virus](/Dataset/human/archiveit/20171231000000_ebola_virus/ebola_virus_qp_mc.json.ss)    |       2014-10-01      |        133        |
|        No        |   Twitter-Top  |  [hurricane harvey](/Dataset/twitter_top/20200412025044_hurricane_harvey/hurricane_harvey_qp_mc.json.mc)  |       2020-04-11      |  201 (500 tweets) |
|        No        |   Twitter-Top  | [flint water crisis](/Dataset/twitter_top/20200410034257_flint_water_crisis/flint_water_crisis_qp_mc.json.mc) |       2020-04-09      |  312 (500 tweets) |
|        No        |   Twitter-Top  |     [coronavirus](/Dataset/twitter_top/20200410154858_coronavirus/coronavirus_qp_mc.json.mc)    |       2020-04-09      |  533 (500 tweets) |
|        No        |   Twitter-Top  |   [2018 world cup](/Dataset/twitter_top/20190109000000_2018_world_cup/2018_world_cup_qp_mc.json.mc)   |       2019-01-09      |  121 (500 tweets) |
|        No        |   Twitter-Top  |     [ebola virus](/Dataset/twitter_top/20171231000000_ebola_virus/ebola_virus_qp_mc.json.mc)    | 2017-(11-30 to 12-31) |   48 (68 tweets)  |
|        No        |   Twitter-Top  |  [hurricane harvey](/Dataset/twitter_top/20170901000000_hurricane_harvey/hurricane_harvey_qp_mc.json.mc)  |   2017-09-(02 to 31)  |  95 (153 tweets)  |
|        No        | Twitter-Latest | [flint water crisis](/Dataset/twitter_latest/20200410034257_flint_water_crisis/flint_water_crisis_qp_mc.json.mc) |       2020-04-09      |  92 (500 tweets)  |
|        No        | Twitter-Latest |     [coronavirus](/Dataset/twitter_latest/20200410154858_coronavirus/coronavirus_qp_mc.json.mc)    |       2020-04-09      |  541 (500 tweets) |
|        No        | Twitter-Latest |   [2018 world cup](/Dataset/twitter_latest/20190109000000_2018_world_cup/2018_world_cup_qp_mc.json.mc)   |       2019-01-09      |  84 (488 tweets)  |
## Code 
Consider the following implementation details for the Quality Proxies. All section references point to the sections in the paper.

### Post popularity (Section 3.2)
The post popularity was instantiated with with metrics that count how many people replied to (`replies rp`), shared (`shares sh`), and liked (`likes lk`) a social media post. All of these are normalized with [Min-max feature scaling](https://en.wikipedia.org/wiki/Feature_scaling) to fit within the `[0, 1]` range. In which 0 represents the least popular post and 1 represents the most popular post.

### Author popularity (Section 3.3)
We instantiated the author-popularity `ap` with the difference between the `in-degree` and the `out-degree` (ap = in-degree − out-degree) normalized. For Twitter, the in-degree represents the number of followers (`followers_count`) and out-degree represents the number of followings (`friends_count`). Both values can be extracted from the [user object](https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/overview/user-object). If the `in-degree < out-degree`, then ap < 0 eventhough we want it to fall within `[0, 1]`. To fix this, the offset (the absolute value of smallest difference between in-degree and out-degree) is added to each difference before [Min-max feature normalization](https://en.wikipedia.org/wiki/Feature_scaling). The table below illustrates an example for calculating the `ap` the following Twitter accounts: `@WHO`, `@IOMSouthSudan`, `@PNASNews`, and `@Microbiology_LR`.

The author-popularity ap<sub>i</sub> values  of  four  seeds.   The `in-` and `out-degree` details were extracted on February 15, 2020.  The `offset = 0` since the minimum d<sub>i</sub>, 3,723 ≥ 0. The difference between the minimum and maximum d<sub>i</sub>, (5,394,414 = 5,398,137 − 3,723) was used to normalize ap<sub>i</sub>. dp<sub>i</sub> is calculated in the same fashion as ap<sub>i</sub> with one important difference: the `in-` and `out-degree` information is extracted from the Twitter handle that has a bi-directional link with the domain of the seed.
|        User        | in-degree (in<sub>i</sub>) | out-degree (out <sub> i</sub>) | d<sub>i</sub> = in<sub>i</sub> - out<sub>i</sub> | d <sub>i</sub> + offset = d<sub>i</sub> + 0 = d<sub>i</sub> | ap<sub>i</sub> |
|:------------------:|:--------------------------:|:------------------------------:|:------------------------------------------------:|-------------------------------------------------------------|:--------------:|
|       `@WHO`       |          5,399,854         |              1,717             |                     5,398,137                    |                          5,398,137                          |        1       |
|  `@IOMSouthSudan`  |            9,237           |               740              |                       8,497                      |                            8,497                            |     0.0008     |
|     `@PNASNews`    |           118,866          |              1,343             |                      117,523                     |                           117,523                           |     0.0210     |
| `@Microbiology_LR` |            3,886           |               163              |                       3,723                      |                            3,723                            |        0       |
|                    |                            |                                |              min(d<sub> i</sub> )              | 3,723                                                       |                |
|                    |                            |                                |              max(d<sub> i</sub> )              | 5,398,137                                                   |                |

### Domain popularity (Section 3.4)
We instantiated the domain-popularity `dp` quality proxy by approximating the popularity of the social media account (e.g., `@CDCgov`) associated withthe seed domain (e.g., `cdc.gov`). To calculate `dp` for a seed (e.g., https://www.cdc.gov/coronavirus/2019-nCoV/)), utilizing Twitteras example, first, we must find the social media account (https://twitter.com/CDCgov) associated with the domain (e.g., `cdc.gov`). This is done by finding a bi-directional link between the social me-dia account and the seed's website. Therefore, 
`dp` is calculated in the same fashion as ap with one important difference: the `in-` and `out-degree` information is extracted from the Twitter handle that has a bi-directional link with the domain of the seed as illustrated by the [Table in Section 3.4](#author-popularity-section-33)

### Geographical (Section 4.1) Quality Proxy
We instantiated author (ge<sub>a</sub>) and domain (ge<sub>d</sub>) QPs with the normalized (`[0, 1]`) distance (measured with the [Haversine formula](/Code/haversine.py)) between a reference epicenter and the geo-location associated with the post author (for ge<sub>a</sub>) or social media account associated via a bi-directional link (similar to `dp`) with the seed domain (for ge<sub>d</sub>). We utilized the [Google Maps Services Places API](https://developers.google.com/places/web-service/search) to normalize names (e.g., "NYC" and "New York") into a single name and geo-coordinates.

The `normalizeLoc()` [function](/Code/normalizeLocation.py) implements the normalization of location:
```
gmaps = googlemaps.Client(key=googlemapsKey)
places = gmaps.places("NYC")
```
The function requires you to get your Google Maps Key. This can be done for free by following the following steps:
1. Created account on [Google Cloud Platform](https://console.cloud.google.com/) and Project
2. Activated Places API from Google Cloud Platform Dashboard
3. Enable billing (You're given free trial credit of $300 for 12 months, subsequently you may pay as you go if you choose to upgrade)

### Temporal (Section 4.2) Quality Proxy
We instantiate the Temporal QP with the normalized time difference between the publication date of the seed and the reference point considered early. [Newspaper](https://github.com/codelucas/newspaper) can be used to extract the publication date from news articles when available within the document. Otherwise [CarbonDate](https://github.com/oduwsdl/CarbonDate) can be used to estimate the creation date of the document.

### Subject-expert (Section 4.3) Quality Proxy
We used Document Frequency (DF) to instantiate the subject-expertise (`su`) of the domain of a seed. We extracted DF scores by counting the number of result pages returned by Google for a given query normalized (divided) by the total number of pages indexed by the search engine for the site.

### Retrievability (Section 4.4) Quality Proxy
We instantiated the Retrievability `rt` of a seed (e.g., https://www.cdc.gov/vhf/ebola/index.html) with its reciprocal rank 1/rank<sub>d</sub> (e.g., 1/2) when searching the first k (e.g., k=20) Google SERPs for the seed with the query (e.g., "ebola virus") used to extract seeds. This might require scraping Google to extract the number of document hits returned by the Google Search Engine Result Page (SERP). We did not provide code for Scraping Google since scraping is not allowed. However, some [libraries](https://github.com/NikolaiT/GoogleScraper) provide this capability.

### Reputation (Section 4.5) Quality Proxy
The broad (re<sub>b</sub>) and narrow (re<sub>n</sub>) Reputation QPs for seed domains where instantiated with the citation rates of seed domains from the references of [gold-standard Wikipedia documents](/Dataset/reputation_gold_standard).

### Relevance (Section 4.6) Quality Proxy
We instantiated the Relevance `rl` QP by simply measuring the cosine-similarity between a seed's document vector and a gold-standard document vector that captures our definition of relevance. The gold-standard is created by concatenating the text of hand-selected documents (Section 8.2, Step 1) that are relevant to a topic, and creating a feature (vocabulary) vector consisting of the TF or TFIDF weights of the terms in the concatenated document.

The documents from the references of the Wikipedia articles below were used to generate document vectors for measuring relevance.  Relevance was approximated by the similarity between a seed's document vector and the gold-standard vector corresponding to the seed's topic.  Similarity exceeding the specified relevance threshold signaled the relevance of the seed.
|        Topic       |                         Wikipedia Reference                        | Relevance Threshold | Document Count |
|:------------------:|:------------------------------------------------------------------:|:-------------------:|:--------------:|
|  hurricane harvey  |           https://en.wikipedia.org/wiki/Hurricane_Harvey           |         0.10        |       183      |
| flint water crisis |          https://en.wikipedia.org/wiki/Flint_water_crisis          |         0.20        |       550      |
|     coronavirus    |           https://en.wikipedia.org/wiki/COVID-19_pandemic          |         0.20        |       719      |
|   2018 world cup   |          https://en.wikipedia.org/wiki/2018_FIFA_World_Cup         |         0.20        |       400      |
|     ebola virus    | https://en.wikipedia.org/wiki/Western_African_Ebola_virus_epidemic |         0.20        |       697      |

### Scarcity (Section 4.7) Quality Proxy
The Scarcity QP requires [extracting the domain](/Code/getDomain.py) of seeds in order to calculate the ratio of the frequency of the domain to the total number of domains.
