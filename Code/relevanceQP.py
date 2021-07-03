import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

def cosineSim(X, Y):
    try:
        X2Norm = np.linalg.norm(X, 2)
        Y2Norm = np.linalg.norm(Y, 2)
        
        if( X2Norm == 0 or Y2Norm == 0 ):
            return 0

        return round(np.dot(X, Y)/(X2Norm * Y2Norm), 10)
    except:
        print("Error, catch it here")
        return 0

def getStopwordsSet(frozenSetFlag=False):
    stopwords = getStopwordsDict()
    
    if( frozenSetFlag ):
        return frozenset(stopwords.keys())
    else:
        return set(stopwords.keys())

def getStopwordsDict():

    stopwordsDict = {
        "a": True,
        "about": True,
        "above": True,
        "across": True,
        "after": True,
        "afterwards": True,
        "again": True,
        "against": True,
        "all": True,
        "almost": True,
        "alone": True,
        "along": True,
        "already": True,
        "also": True,
        "although": True,
        "always": True,
        "am": True,
        "among": True,
        "amongst": True,
        "amoungst": True,
        "amount": True,
        "an": True,
        "and": True,
        "another": True,
        "any": True,
        "anyhow": True,
        "anyone": True,
        "anything": True,
        "anyway": True,
        "anywhere": True,
        "are": True,
        "around": True,
        "as": True,
        "at": True,
        "back": True,
        "be": True,
        "became": True,
        "because": True,
        "become": True,
        "becomes": True,
        "becoming": True,
        "been": True,
        "before": True,
        "beforehand": True,
        "behind": True,
        "being": True,
        "below": True,
        "beside": True,
        "besides": True,
        "between": True,
        "beyond": True,
        "both": True,
        "but": True,
        "by": True,
        "can": True,
        "can\'t": True,
        "cannot": True,
        "cant": True,
        "co": True,
        "could not": True,
        "could": True,
        "couldn\'t": True,
        "couldnt": True,
        "de": True,
        "describe": True,
        "detail": True,
        "did": True,
        "do": True,
        "does": True,
        "doing": True,
        "done": True,
        "due": True,
        "during": True,
        "e.g": True,
        "e.g.": True,
        "e.g.,": True,
        "each": True,
        "eg": True,
        "either": True,
        "else": True,
        "elsewhere": True,
        "enough": True,
        "etc": True,
        "etc.": True,
        "even though": True,
        "ever": True,
        "every": True,
        "everyone": True,
        "everything": True,
        "everywhere": True,
        "except": True,
        "for": True,
        "former": True,
        "formerly": True,
        "from": True,
        "further": True,
        "get": True,
        "go": True,
        "had": True,
        "has not": True,
        "has": True,
        "hasn\'t": True,
        "hasnt": True,
        "have": True,
        "having": True,
        "he": True,
        "hence": True,
        "her": True,
        "here": True,
        "hereafter": True,
        "hereby": True,
        "herein": True,
        "hereupon": True,
        "hers": True,
        "herself": True,
        "him": True,
        "himself": True,
        "his": True,
        "how": True,
        "however": True,
        "i": True,
        "ie": True,
        "i.e": True,
        "i.e.": True,
        "if": True,
        "in": True,
        "inc": True,
        "inc.": True,
        "indeed": True,
        "into": True,
        "is": True,
        "it": True,
        "its": True,
        "it's": True,
        "itself": True,
        "just": True,
        "keep": True,
        "latter": True,
        "latterly": True,
        "less": True,
        "made": True,
        "make": True,
        "may": True,
        "me": True,
        "meanwhile": True,
        "might": True,
        "mine": True,
        "more": True,
        "moreover": True,
        "most": True,
        "mostly": True,
        "move": True,
        "must": True,
        "my": True,
        "myself": True,
        "namely": True,
        "neither": True,
        "never": True,
        "nevertheless": True,
        "next": True,
        "no": True,
        "nobody": True,
        "none": True,
        "noone": True,
        "nor": True,
        "not": True,
        "nothing": True,
        "now": True,
        "nowhere": True,
        "of": True,
        "off": True,
        "often": True,
        "on": True,
        "once": True,
        "one": True,
        "only": True,
        "onto": True,
        "or": True,
        "other": True,
        "others": True,
        "otherwise": True,
        "our": True,
        "ours": True,
        "ourselves": True,
        "out": True,
        "over": True,
        "own": True,
        "part": True,
        "per": True,
        "perhaps": True,
        "please": True,
        "put": True,
        "rather": True,
        "re": True,
        "same": True,
        "see": True,
        "seem": True,
        "seemed": True,
        "seeming": True,
        "seems": True,
        "several": True,
        "she": True,
        "should": True,
        "show": True,
        "side": True,
        "since": True,
        "sincere": True,
        "so": True,
        "some": True,
        "somehow": True,
        "someone": True,
        "something": True,
        "sometime": True,
        "sometimes": True,
        "somewhere": True,
        "still": True,
        "such": True,
        "take": True,
        "than": True,
        "that": True,
        "the": True,
        "their": True,
        "theirs": True,
        "them": True,
        "themselves": True,
        "then": True,
        "thence": True,
        "there": True,
        "thereafter": True,
        "thereby": True,
        "therefore": True,
        "therein": True,
        "thereupon": True,
        "these": True,
        "they": True,
        "this": True,
        "those": True,
        "though": True,
        "through": True,
        "throughout": True,
        "thru": True,
        "thus": True,
        "to": True,
        "together": True,
        "too": True,
        "toward": True,
        "towards": True,
        "un": True,
        "until": True,
        "upon": True,
        "us": True,
        "very": True,
        "via": True,
        "was": True,
        "we": True,
        "well": True,
        "were": True,
        "what": True,
        "whatever": True,
        "when": True,
        "whence": True,
        "whenever": True,
        "where": True,
        "whereafter": True,
        "whereas": True,
        "whereby": True,
        "wherein": True,
        "whereupon": True,
        "wherever": True,
        "whether": True,
        "which": True,
        "while": True,
        "whither": True,
        "who": True,
        "whoever": True,
        "whole": True,
        "whom": True,
        "whose": True,
        "why": True,
        "will": True,
        "with": True,
        "within": True,
        "without": True,
        "would": True,
        "yet": True,
        "you": True,
        "your": True,
        "yours": True,
        "yourself": True,
        "yourselves": True
    }
    
    return stopwordsDict

def getTFMatrixFromDocList(oldDocList, params=None):

    if( len(oldDocList) == 0 ):
        return {}

    docList = [ d for d in oldDocList if len(d) != 0 ]
    if( len(docList) == 0 ):
        return {}

    if( params is None ):
        params = {}

    params.setdefault('idf', False)
    params.setdefault('norm', 'l2')#see TfidfTransformer for options

    params.setdefault('normalize', False)#normalize TF by vector norm (L2 norm)
    params.setdefault('ngram_range', (1, 1))#normalize TF by vector norm (L2 norm)
    params.setdefault('tokenizer', None)
    params.setdefault('verbose', False)
    params.setdefault('no_ngram_freqs', False)
    params.setdefault('token_pattern', r'(?u)\b[a-zA-Z\'\’-]+[a-zA-Z]+\b|\d+[.,]?\d*')
            

    count_vectorizer = CountVectorizer(token_pattern=params['token_pattern'], tokenizer=params['tokenizer'], stop_words=getStopwordsSet(), ngram_range=params['ngram_range'])
    tf_mat = count_vectorizer.fit_transform(docList).toarray()
    payload = {}

    if( params['normalize'] is True  ):
        tf_mat = normalize(tf_mat, norm=params['norm'], axis=1)

    elif( params['idf'] is True ):
        tfidf = TfidfTransformer( norm=params['norm'] )
        tfidf.fit(tf_mat)
        tf_idf_matrix = tfidf.transform(tf_mat).todense()
        payload['tf_idf_matrix'] = tf_idf_matrix

    payload['tf_mat'] = tf_mat
    
    if( params['no_ngram_freqs'] is False ):

        top_freq_ngrams = []
        vocab = count_vectorizer.get_feature_names()
        all_col_sums_tf = np.sum(tf_mat, axis=0)
        
        for i in range( len(vocab) ):
            top_freq_ngrams.append( {'term': vocab[i], 'tf': int(all_col_sums_tf[i])} )
        
        payload['ngram_freqs'] = sorted(top_freq_ngrams, key=lambda x: x['tf'], reverse=True)
         
    
    if( params['verbose'] is True ):
        np.set_printoptions(threshold=sys.maxsize, linewidth=100)
        print('\nVOCABULARY')
        print( count_vectorizer.get_feature_names() )

        print('\nDENSE tf_mat matrix')
        print( payload['tf_mat'] )
        
        if( 'ngram_freqs' in payload ):
            print('\nngram_freqs')
            print( payload['ngram_freqs'] )
        
        if( 'tf_idf_matrix' in payload ):
            print('\nDENSE tf_idf_matrix matrix')
            print( payload['tf_idf_matrix'] )
    
    
    payload['tf_mat'] = payload['tf_mat'].tolist()
    if( 'tf_idf_matrix' in payload ):
        payload['tf_idf_matrix'] = payload['tf_idf_matrix'].tolist()

    return payload

def calcPairSim(matrix):
        
    if( len(matrix) != 2 ):
        return -1

    params = {}
    params['normalize'] = True
    matrix = getTFMatrixFromDocList( matrix, params=params )

    if('tf_mat' not in matrix ):
        return -1

    matrix = matrix['tf_mat']
    if( len(matrix) != 2 ):
        return -1
    
    return cosineSim(matrix[0], matrix[1])

doc_0 = '''
Ecological epidemiology
Abstract
Ebola virus disease (EVD) is a contagious, severe and often lethal form of hemorrhagic fever in humans. The association of EVD outbreaks with forest clearance has been suggested previously but many aspects remained uncharacterized. We used remote sensing techniques to investigate the association between deforestation in time and space, with EVD outbreaks in Central and West Africa. Favorability modeling, centered on 27 EVD outbreak sites and 280 comparable control sites, revealed that outbreaks located along the limits of the rainforest biome were significantly associated with forest losses within the previous 2 years. This association was strongest for closed forests (>83%), both intact and disturbed, of a range of tree heights (5–>19 m). Our results suggest that the increased probability of an EVD outbreak occurring in a site is linked to recent deforestation events, and that preventing the loss of forests could reduce the likelihood of future outbreaks.
Introduction
Ebola virus disease (EVD) is a zoonosis that causes severe and often fatal haemorrhagic fever in humans 1 . EVD was first identified in Africa in 1976 2 and since then is estimated to have killed over 13,000 people 3 . Due to its associated high mortality and potential for contagion, EVD is viewed as a global threat 4 . EVD is propagated by a group of filovirus species of the genus Ebolavirus (hereafter Ebola virus) 1 , but despite advances in understanding this zoonotic disease, the factors that trigger and maintain outbreaks remain elusive 5 . Such uncertainties impede the more accurate and effective prediction of outbreaks that would facilitate improved response or prevention 6 . Human activities may have promoted direct or indirect contact between humans and an animal reservoir of the virus 7 . Some suggest that the loss of forest can facilitate the spread of the disease to non-forest areas 8 , 9 . The mechanism, although unknown, likely results from more frequent contact between infected wild animals and humans. The latest outbreak in Guinea has been linked to contact with a bat colony, an event that some have linked to forest loss 10 , 11 . However, the enabling role of forest loss in Ebola outbreaks seems hard to reconcile with the upper Guinea forests having been a dynamic mosaic of forest, savannah, and farmland for centuries, and that people in this region have been sympatric with bats, and other forest wildlife, throughout this history 12 . More generally, humans 13 and great apes 14 have lived in close proximity to bats for millennia, thus it may be simplistic to claim that forest loss was sufficient to cause the emergence of EVD and its repeated outbreaks.
Initial suggestions that deforestation increases zoonotic EVD outbreaks result from observations in seven West African EVD outbreak sites that revealed greater forest fragmentation in these locations than in their surroundings 15 . Quantitative analysis of the nexus between deforestation and the emergence of Ebola virus disease has been recently undertaken by Rulli et al. 16 . Although this study showed that EVD outbreaks occurred mostly in hotspots of forest fragmentation, the spatio-temporal dynamics of this relationship was not considered. Here, using forest change remote sensing data 17 and modeling we investigate: (1) the spatio-temporal association between forest types and forest changes and the possibility of an outbreak, and (2) whether this association can be extended to the whole distribution range of the Ebola virus in West and Central Africa 18 .
In this study, our original hypothesis was that deforestation leads to increased contacts between humans and hosts or vulnerable mammals and thus leads to zoonotic outbreaks of EVD. The underlying assumption is that zoonotic transmission would be more probable either because hunters would travel further (or therefore increase possible contacts) or because in fragmented forests there can be an increased density of bats and other potential reservoirs of Ebola virus 7 , 8 , 9 . In order to test these relationships, we investigated vegetation-cover changes within human populated areas where Ebola outbreaks have occurred and compared these to otherwise comparable localities where there have not been outbreaks. Out of the 40 EVD outbreaks reported since 1976 we were able to focus on 27 sites where index cases (the first patient that indicates the existence of an EVD outbreak) were identified (Table  S1 ) and for which large-scale deforestation data were available for the period 2001–2014 17 . We used spatial distribution modeling based on the Favorability Function 19 to develop spatio-temporal distribution models using a set of predictor variables (Table  S2 ) to discriminate between EVD outbreak locations and 280 control spatio-temporal locations (randomly selected sites containing human settlements but no recorded EVD outbreaks) (Figure  S1 ). We assessed annual forest loss and fragmentation within a 20-km radius buffer around each outbreak and non-outbreak site (See Methods); the buffer radius reflecting the distance hunters typically range from their villages 20 , 21 and thus defines the area where we assume zoonotic disease transmission could occur.
Results and Discussion
The spatio-temporal (STP) model detected a significant (χ2 = 55.286; p = 2.74 × 10−8) spatio-temporal trend in which 16 of the 17 EVD outbreaks during 2001–2005 were clustered around the Gabon-Republic of Congo border, in the western Congo basin. After 2006, 8 of the 9 outbreaks occurred in Uganda and eastern/southern Democratic Republic of Congo (DRC), and after 2013 in West Africa (Fig.  1A , Tables  S3 and S4 ).
Figure 1
3D representations of two models of favorability (F) for the occurrence of EVD outbreaks. (A) Model describing the spatio-temporal pattern (STP) of outbreaks between 2001 and 2014. (B) Model based on forest loss (FL) for the period 2006 and 2014. Yellow points indicate outbreak locations; red points represent random locations with no report of EVD outbreaks. Compared to random locations, favorability based on FL is significantly higher for seven of the nine outbreaks that occurred after 2005. The map of Africa shows the area represented by axes x and y; country borders are outlined in grey; the green area represents rainforests (GlobCover version 2.1 database for 2005–2006, © ESA/ESA Globcover 2005 Project, led by MEDIAS-France/POSTEL).
Full size image
Importantly, the 6 forest loss (FL) models only selected variables indicating forest changes less than 3 years prior to the outbreaks (Table  S3 ). In all cases, higher proportions of dense forest losses significantly favored later EVD outbreaks. The most significant FL model (χ2 = 25.925; p = 0.000033), and with the highest capacity to discriminate between presences and absences of EVD outbreaks (AUC = 0.910), was that generated for the period 2006–2014 (Fig.  2 , Table  S4 ). This model indicated favorable conditions (F > 0.5) for 7 of the 9 EVD outbreaks in this period —those in Uganda, DRC and Guinea, all close to the limits of the rainforest biome (Fig.  1B ) — and low favorability (F < 0.5) for 89% of non-outbreak locations. The variable most significantly associated with outbreaks, according to this model, was the proportion of ‘not intact forest with dense cover (>83%) and tall trees (>19 m)’ lost in the buffer area the same year as the outbreak occurred (Wald = 7.421; p = 0.006) (Figures  S2 and S3 , Table  S3 ). The densest canopy cover (>83%) characterized 3 out of the 4 variables in the model. However, both intact and previously disturbed forests, and a wide range of tree heights (5–>19 m) were represented in the final model.
Figure 2
Capacity of discrimination (Area Under the Curve, AUC) and classification [sensitivity, specificity, Correct Classification Rate (CCR) and Kappa] of the six models defining the favorability of the occurrence of EVD outbreaks due to forest loss (FL). The x-axis is the starting year of the time period considered in the models.
Full size image
The FL models that were run considering periods 2003–2014, 2004–2014 and 2005–2014 were sensitive to the 7 outbreaks explained by the 2006–2014 model, but they did not provide any evidence for a relation between forest loss and outbreaks prior to 2006. Only the FL model for 2002–2014 (χ2 = 20.7; p = 0.000114) described favorable conditions for outbreaks that occurred before that year, although with less overall discrimination (AUC = 0.755) and classification power than the 2006–2014 model (Fig.  2 , Table  S4 ). Because of this, both 2002–2014 and 2006–2014 time periods were further considered for variation partitioning analyses (See Supplementary Methods).
Significant basal spatial favorability (BSF) models were only found for the periods 2001–2014, 2002–2014 and 2006–2014. In the two former periods, there was a significant association (χ2 > 5.000; p < 0.025) between favorability for the Ebola virus and outbreaks. This supports that differences in environmental potential for Ebola virus presence in the environment were related to the probability of occurrence of EVD outbreaks (Fig.  3 , Table  S3 ). However, this does not apply to the period 2006–2014, when outbreaks —which occurred along the limits of the West and Central African rainforest distribution range— were moderately associated to denser human populations (χ2 = 2.771; p = 0.096). Nevertheless, the discrimination and classification capacities of the BSF model between 2006 and 2014 were not as good (AUC = 0.812; sensitivity = 0.556; kappa = 0.175) as those based on forest loss (AUC = 0.910; sensitivity = 0.778; kappa = 0.341) (Table  S4 ).
Figure 3
Models based on the basal spatial favorability (BSF) for EVD. Point colors indicate favorability (ranging 0–1). (A) The explanatory variable in the model for the period 2002–2014 is the environmental/zoogeographical favorability for the occurrence of Ebola virus in the wild (12). (B) The explanatory variable in the model for the period 2006–2014 is the rural human population density. Maps were generated using ArcGIS 10.3 ( http://desktop.arcgis.com/en/ ).
Full size image
Variation partitioning analyses regarding the complete Ebola virus area 18 (EVA) revealed that, in the period 2002–2014, the contribution made by forest loss (FL) alone to explain the variation in favorability for EVD outbreak was low (4.35%) compared with that made by the spatio-temporal pattern (STP) (82.44%) (Fig.  4 ). However, the importance of forest loss increased dramatically to 26.90% for the period 2006–2014. In the area within the range of favorability values overlapping with EVD outbreaks (EVDA), forest loss alone accounted for 6.88% of the variation (compared with 72.86% for the STP) in 2002–2014, but increased to 59.93% (vs. only 6.49% for the STP) in 2006–2014. From 2006 to 2014, the basal spatial favorability (BSF) explained a low proportion of the EVD outbreaks in EVA (4.74%), while gaining relevance in EVDA (12.95%), but even then, forest loss alone was 4.6 times more relevant (59.93%) than the contribution of BSF alone.
Figure 4
Venn diagrams displaying the results of variation partitioning analyses of a model combining spatio-temporal pattern (STP), forest loss (FL) and basal spatial favorability (BSF) for EVD outbreaks, for the periods 2002–2014 and 2006–2014. (A) Analysis focused on the complete Ebola-virus area (EVA) 18 . (B) Analysis focused on the range of favorability values overlapping with EVD outbreaks (EVDA).
Full size image
Our study independently supports recent findings of an association between EVD outbreak locations and forest loss after 2004 16 . In addition, the spatio-temporal approach and the inclusion of all outbreaks later than 2000 revealed that: (1) the EVD outbreak-forest loss link is only significant around the limits of the West and Central African rainforest biome and excludes other EVD areas in the western Congo basin; (2) there is a time lag between forest loss and EVD outbreaks of 2 years; and (3) the loss of dense forests, principally those with >83% canopy cover, is an important factor. We also show that zoonotic EVD outbreaks appear in areas where human population density is high and where the virus has favorable conditions, but the relative importance of forest loss is partially (>60%) independent of these factors (Fig.  4 ).
The coupling between EVD outbreaks and forest loss in the margins of the rainforest biome within the previous two years, highlighted in our study, has profound implications. A plausible explanation is that contact between humans and infected wildlife increases dramatically after the removal of forest. Such an effect has been previously suggested 22 , 23 , and while our results strongly support such an interpretation, they also indicate that the changes are not sustained beyond two years. A variety of ecological descriptors (e.g. species richness) are affected soon after forest fragmentation 24 , and the factors promoting the emergence of the Ebola virus (host range, reservoir species, circulation in nature) are still unknown 5 , 21 . Forest loss disrupts animal movements and local densities, and thus influences their interactions and the potential for a pathogen to be transmitted between individuals and across species —though for Ebola such mechanisms remain theoretical. Regardless of whether or not fruit bats are important reservoirs of Ebola virus 6 , these animals are evidently involved in the virus’ ecology 25 , 26 . Deforestation influences fruit bat movement and abundance 23 , 27 , 28 , and the composition, abundance and behaviors of the wider mammal fauna is influenced by timber cutting and disturbance 29 . Thus, forest loss and fragmentation could favor the combination of ecological events that are required for viral emergence. Interestingly, our results, which are not limited to tall intact old growth forests, highlight the association between EVD outbreaks and close-canopy forests.
EVD outbreaks may increase in the coming decades. Human population growth, and greater penetration into Africa’s remaining forests will be coupled with the proliferation of potential reservoir species as urbanization, agriculture and deforestation intensify 30 , 31 , 32 . Greater population mobility via improved roads and air access increases the risks of an undetected EVD outbreak becoming a pandemic. A rapid response to any outbreak is fundamental to reducing contagion and reducing mortality, but increased preparation and vigilance are needed to diminish the risk. The challenge, however, is enormous. The vast and still relatively inaccessible areas involved, alongside the limited resources and manpower available, make it clear that priority setting will better guide monitoring and ensure preparedness. Health services in frontier areas require bolstering but the need for interdisciplinary approaches to improve our understanding of the ecology of the virus and its hosts, as we suggest in our study, cannot be neglected 30 . Our results provide indicators that could be useful for predicting where and when EVD outbreaks are more likely to occur. The data availability and image processing requirements of rapid predictions appear well within current technical abilities, though the best way to update forest cover classifications, or measures, requires further evaluation (to balance feasibility and utility). Such an approach would consider inhabited areas around the limits of the West and Central African rainforest biome favorable for Ebola, and identify where dense cover forests has been lost in the previous two years. This predictive system would draw special attention to high-risk locations, which can be updated and improved as data and concepts advance. For example, our analyses do not reveal why EVD outbreaks prior to 2005 occurred deep in the rainforest biome —within Republic of Congo and Gabon— but subsequently shifted to more transitional forests —in Uganda, DRC and Guinea. These regional shifts could result from inter-annual variation in rainfall across the continent 33 , 34 and may reflect fruit availability and associated movements though more data would be needed to clarify these putative relationships.
Prevention of EVD outbreaks is the ultimate goal. Accepting the inferred links to land cover as causal implies that the risk of zoonotic EVD outbreaks can be diminished by (1) reducing deforestation and (2) reducing human proximity and access to recently damaged forests (for two years). More generally, our results show that forest loss, like EVD, should be seen as a major global health issue and should be managed and funded accordingly.
Methods
Vegetation Cover
We selected 27 locations where Ebola virus disease (EVD) outbreak “index cases” (i.e. also primary cases or patients zero, the initial identified patients in specific outbreaks assumed to result from zoonotic transmission) had occurred (Table  S1 ), and 280 locations without outbreaks as control areas (see criteria for buffer selection below in the section “Deforestation”). We evaluated vegetation-cover changes within a buffer area of radius 20-km around each location selected. Based on past work in West and Central Africa we judged 20 km the distance that could typically be traveled by hunters 20 , 21 . Within these buffers we described vegetation cover changes as a set of dependent variables used in our models (Table  S2 ). The use of non-overlapping buffers prevented the possible problems arising from spatial autocorrelation among the data.
Forest loss was estimated from data of vegetation cover changes contained in the University of Maryland’s Global Forest Change (GFC) project 17 . The GFC dataset is a time-series analysis of Landsat images characterizing forest extent and change at a 30-meter spatial resolution. This dataset reports annual forest extent, loss, and gain from the period 2001–2014. Because trees are simply defined as vegetation taller than 5-m in height and with a canopy cover of more than 25%, losses of natural forest or planted vegetation are not distinguishable using GFC data. Given this constraint, we applied the forest cover classification developed by Tyukavina et al. 35 , who stratified forests according to 7 distinct classes, based on canopy structure, as defined by percent cover, height, and intactness according to Potapov’s et al. 36 description of intact forest landscapes (IFL, defined as an unbroken expanse of natural ecosystems within the zone of current forest extent, showing no signs of significant human activity, and large enough that all native biodiversity, including viable populations of wide-ranging species, could be maintained) 36 :
1.
Forest with low cover (between 25–45% canopy cover).
2.
Forest with medium cover and short trees (45–83% canopy cover, 5 to 11-m height).
3.
Forest with medium cover and tall trees (between 45–83% canopy cover, ≥11-m height).
4.
Not IFL with dense cover and short trees (>83% canopy cover, <19-m height).
5.
IFL with dense cover and short trees (with no signs of human disturbance, >83% canopy cover, <19-m height).
6.
Not IFL with dense cover and tall trees (>83% canopy cover, ≥19-m height).
7.
IFL with dense cover and tall trees (pristine old-growth natural forests, >83% canopy cover, ≥19-m height).
Forest area loss within each 20-km radius buffer area was estimated annually for the period 2001–2014. This estimation was made in ha (i.e. absolute forest loss, AFL), and also in percentage respect to year 2000 (i.e. relative forest loss, RFL). In order to characterize the type of forest to which quantified tree-losses were related, we made all calculations with reference to these forest strata:
1.
Total forest (sum of strata 1 to 7).
2.
Dense forest (sum of strata 4 to 7).
3.
IFL (sum of strata 5 and 7).
4.
Every forest stratum (1 to 7) separately.
We developed a second set of variables to describe patterns of forest fragmentation (Table  S2 ), again employing data from the Global Forest Change (GFC) project 17 :
1.
Mean distance to forest edge (MDFE) 24 : To quantify the grade of forest fragmentation, we calculated the average distance between every forested pixel within each 20-km radius buffer area and its nearest non-forest pixel. This variable was conceived as a spatial, not temporal characteristic of landscape, and so the calculations were referred to a single time period, for which we selected year 2001 (i.e. the date of the first EVD outbreak considered).
2.
Increased edge (IE) 24 : This variable describes the change in relative length of the limit between forest and non-forest. We followed three steps: (1) calculate the length of forest edge in 2000 (i.e. the year before the study period) and in 2014 (i.e. the end of the study period); (2) calculate the increase in length between 2000 and 2014 (i.e. length in 2014 minus length in 2000); (3) calculate the proportion of increase with respect to the length in 2000.
MDFE and IE were calculated for total forest, for dense forest and for IFL.
Deforestation
Our dataset included the locations and dates of 28 EVD outbreak “index cases” since 2001 (Table  S1 ). Two outbreaks in Ekata (Gabon) occurred in the same year and were merged for our analysis; thus, operatively, the number of outbreak locations considered was 27.
Deforestation data were available from 2001 to 2014. Thus, given that our working hypothesis was that EVD outbreaks might occur as a consequence of deforestation, only cases within this period were considered.
In the search for ecologically meaningful relationships between deforestation and outbreaks, we needed to analyze events within both spatial and temporal contexts. We examined the geographic overlap of deforestation and EVD outbreaks and the time-scale involved. Six time lags after deforestation events were examined: 0 to 5 years. As the outbreaks considered in the study included all index cases between 2001 and 2014, it was not possible to examine time lags higher than 0 in the outbreaks of 2001, time lags higher than 1 in those of 2001 and 2002, and so on until 2006, for which time lags from 0 to 5 years were available in the data set. Because of this, the associations between outbreaks and forest loss were investigated by means of six temporally nested analyses, comprising from 9 to 14 years, all of them ending in 2014 (Table  S5 ).
For our comparisons we selected 280 control locations where outbreaks have never occurred (Figure  S1 ). All these locations were assigned random coordinates in three axes: latitude, longitude and time, considering the period 2001 and 2014. The geographical context was West and Central Africa.
Locations where EVD outbreaks have occurred, as well as the 280 randomly selected non-outbreak control locations were assessed for forest cover change. The random selection of non-outbreak locations was, however, subject to the following constraints:
1.
All such locations were favorable for Ebola virus in wildlife 18 .
2.
A human population was present, according to at least one of the following sources:
Global Rural-Urban Mapping Project, Version 1 (GRUMPv1), Settlement Points (Palisades, NY: NASA Socioeconomic Data and Applications Center SEDAC) http://dx.doi.org/10.7927/H4M906KR .
Interactive Forest Atlas of Cameroon, CAR, Congo, DRC, Equatorial Guinea and Gabon, http://www.wri.org/our-work/project/congo-basin-forests .
Country, Cities and Places GIS Shapefile Map Layers, http://www.mapcruzin.com/ .
The Humanitarian Data Exchange (HDX) https://data.hdx.rwlabs.org/ , NGIS Country Files http://geonames.nga.mil/gns/html/namefiles.html .
Because all our EVD index cases were in rural areas, we restricted our selection of control sites to such context. For this aim, we excluded human populations less than 20-km far from urban areas as the MODIS 500-m Map of Global Urban Extent defines them.
3.
These locations were more than 40 km far from each other and from EVD outbreak locations, so that 20-km radius buffers did not overlap.
The final set of locations in the analysis consisted of 27 circles around EVD outbreak sites, and another 280 similar areas corresponding to non-outbreak sites.
Modelling
The Favorability Function (F, whose range is 0–1) can be defined by both equations  1 and 2 19 , 37 :
F
''' 

doc_1 = '''
Pardis Sabeti and Stephen Gire in the Genomics Platform of the Broad Institute of M.I.T. and Harvard, in Cambridge, Massachusetts. They have been working to sequence Ebola’s genome and track its mutations.
Photograph by Dan Winters
The most dangerous outbreak of an emerging infectious disease since the appearance of H.I.V., in the early nineteen-eighties, seems to have begun on December 6, 2013, in the village of Meliandou, in Guinea, in West Africa, with the death of a two-year-old boy who was suffering from diarrhea and a fever. We now know that he was infected with Ebola virus. The virus is a parasite that lives, normally, in some as yet unidentified creature in the ecosystems of equatorial Africa. This creature is the natural host of Ebola; it could be a type of fruit bat, or some small animal that lives on the body of a bat—possibly a bloodsucking insect, a tick, or a mite.
Before now, Ebola had caused a number of small, vicious outbreaks in central and eastern Africa. Doctors and other health workers were able to control the outbreaks quickly, and a belief developed in the medical and scientific communities that Ebola was not much of a threat. The virus is spread only through direct contact with blood and bodily fluids, and it didn’t seem to be mutating in any significant way.
After Ebola infected the boy, it went from him to his mother, who died, to his three-year-old sister, who died, and to their grandmother, who died, and then it left the village and began moving through the human population of Guinea, Liberia, and Sierra Leone. Since there is no vaccine against or cure for the disease caused by Ebola virus, the only way to stop it is to break the chains of infection. Health workers must identify people who are infected and isolate them, then monitor everybody with whom those people have come in contact, to make sure the virus doesn’t jump to somebody else and start a new chain. Doctors and other health workers in West Africa have lost track of the chains. Too many people are sick, and more than two hundred medical workers have died. Health authorities in Europe and the United States seem equipped to prevent Ebola from starting uncontrolled chains of infection in those regions, but they worry about what could happen if Ebola got into a city like Lagos, in Nigeria, or Kolkata, in India. The number of people who are currently sick with Ebola is unknown, but almost nine thousand cases, including forty-five hundred deaths, have been reported so far, with the number of cases doubling about every three weeks. The virus seems to have gone far beyond the threshold of outbreak and ignited an epidemic.
The virus is extremely infectious. Experiments suggest that if one particle of Ebola enters a person’s bloodstream it can cause a fatal infection. This may explain why many of the medical workers who came down with Ebola couldn’t remember making any mistakes that might have exposed them. One common route of entry is thought to be the wet membrane on the inner surface of the eyelid, which a person might touch with a contaminated fingertip. The virus is believed to be transmitted, in particular, through contact with sweat and blood, which contain high concentrations of Ebola particles. People with Ebola sweat profusely, and in some instances they have internal hemorrhages, along with effusions of vomit and diarrhea containing blood.
Despite its ferocity in humans, Ebola is a life-form of mysterious simplicity. A particle of Ebola is made of only six structural proteins, locked together to become an object that resembles a strand of cooked spaghetti. An Ebola particle is only around eighty nanometres wide and a thousand nanometres long. If it were the size of a piece of spaghetti, then a human hair would be about twelve feet in diameter and would resemble the trunk of a giant redwood tree.
Once an Ebola particle enters the bloodstream, it drifts until it sticks to a cell. The particle is pulled inside the cell, where it takes control of the cell’s machinery and causes the cell to start making copies of it. Most viruses use the cells of specific tissues to copy themselves. For example, many cold viruses replicate in the sinuses and the throat. Ebola attacks many of the tissues of the body at once, except for the skeletal muscles and the bones. It has a special affinity for the cells lining the blood vessels, particularly in the liver. After about eighteen hours, the infected cell is releasing thousands of new Ebola particles, which sprout from the cell in threads, until the cell has the appearance of a ball of tangled yarn. The particles detach and are carried through the bloodstream, and begin attaching themselves to more cells, everywhere in the body. The infected cells begin spewing out vast numbers of Ebola particles, which infect more cells, until the virus reaches a crescendo of amplification. The infected cells die, which leads to the destruction of tissues throughout the body. This may account for the extreme pain that Ebola victims experience. Multiple organs fail, and the patient goes into a sudden, steep decline that ends in death. In a fatal case, a droplet of blood the size of the “o” in this text could easily contain a hundred million particles of Ebola virus.
Inside each Ebola particle is a tube made of coiled proteins, which runs the length of the particle, like an inner sleeve. Viewed with an electron microscope, the sleeve has a knurled look. Like the rest of the particle, the sleeve has been shaped by the forces of natural selection working over long stretches of time. Ebola is a filovirus, and filoviruses appear to have been around in some form for millions of years. Within the inner sleeve of an Ebola particle, invisible even to a powerful microscope, is a strand of RNA, the molecule that contains the virus’s genetic code, or genome. The code is contained in nucleotide bases, or letters, of the RNA. These letters, ordered in their proper sequence, make up the complete set of instructions that enables the virus to make copies of itself. A sample of the Ebola now raging in West Africa has, by recent count, 18,959 letters of code in its genome; this is a small genome, by the measure of living things. Viruses like Ebola, which use RNA for their genetic code, are prone to making errors in the code as they multiply; these are called mutations. Right now, the virus’s code is changing. As Ebola enters a deepening relationship with the human species, the question of how it is mutating has significance for every person on earth.
The Kenema Government Hospital, in Kenema, Sierra Leone, is a scatter of low yellow-and-red-painted cinder-block buildings with rusty metal roofs. It spreads down a hillside near the center of town, and, according to medical workers there, is normally bustling with patients and their families. The town sits in fertile, hilly country, dotted with small villages, ninety miles southwest of the place where the borders of Sierra Leone, Guinea, and Liberia converge in a triskelion. This border area was the cradle of the Ebola outbreak. For decades, the Kenema hospital has had a special twelve-bed unit called the Lassa Fever Ward and Research Program. Lassa fever is caused by Lassa virus, which is classified by virologists as a Biosafety Level 4 pathogen—lethal, infectious, typically with no vaccine and no reliable cure. In May of this year, the chief physician of the Lassa program, Sheik Humarr Khan, was watching out for Ebola, which, like Lassa, is a Level 4 pathogen. The virus had been spreading in Guinea and Liberia, but there had been no reported cases yet in Sierra Leone.
Around May 23rd, a woman who was having a miscarriage arrived at the hospital. She tested negative for Lassa, but Khan suspected that she might have Ebola. As it turned out, she had been at the funeral of a faith healer who had recently been to Guinea and had died after attempting to heal a number of people sick with Ebola. Khan ordered a blood sample to be taken from her, and he isolated her in the hospital’s Lassa ward. Khan was a specialist in viral hemorrhagic diseases and one of the world’s leading experts in Lassa fever, and people described him as voluble and intense; virus experts from a number of American research institutions had developed close friendships with him and his staff. He devoted much of his time to tending patients at the hospital, who were typically poor. Quite a few of them couldn’t afford to buy medicine, so Khan bought it for them, and he gave them food if they looked hungry. “You must eat or you cannot get better,” he told them.
“I still compose my tweets in longhand on a yellow legal pad.”
When Khan was with patients in the Lassa ward, he wore a type of biohazard outfit known as personal protective equipment, or P.P.E. At Kenema, the outfit consisted of a full-body suit and head covering made of white Tyvek fabric, a breathing mask, a plastic face shield and goggles, two pairs of surgical gloves, one pair of rubber gloves, rubber boots, and a plastic apron. Patients with Lassa had seizures and hemorrhages and went into comas, and many of them died, despite excellent care. In the evening, Khan liked to watch soccer games on television with friends, and when he got tired on his rounds he would sit in a plastic chair for a moment, chatting with people as he drank a can of Sprite.
The day after the woman who had miscarried was admitted to the Lassa unit, a lab technician put on P.P.E., carried a sample of the woman’s blood into the lab, and tested it. It was positive for Ebola. Wanting to be sure, the technician e-mailed the test results to the lab of an associate professor of biology at Harvard University named Pardis Sabeti. Over the years, Sabeti had forged ties with the Lassa program, and had become friends with Khan.
Sabeti is a slender woman in her late thirties, with a warm manner. She is the head of a lab at Harvard, and leads viral-genome efforts at the Broad Institute of M.I.T. and Harvard. She specializes in reading and analyzing the genomes of organisms and, in particular, studies virus evolution—the way viruses change over time as they adapt to their environments. In her spare time, Sabeti is the lead singer and songwriter for an indie band called Thousand Days. Its fourth album has been delayed owing to her work on the Ebola outbreak.
When Sabeti learned that Ebola had reached Sierra Leone, she called a meeting in what she and her colleagues had begun to refer to as the Ebola War Room. It is a sunlit room with a large table at the Broad Institute, on the M.I.T. campus. As the outbreak gathered strength, Sabeti became the de-facto head of a team of scientists who met regularly in the War Room to plan and direct elements of the human defense against Ebola. They had sent team members with advanced diagnostic equipment to Kenema and to Nigeria, to help doctors diagnose Ebola quickly. “The faster you can get a diagnosis of Ebola, the faster you can stop it,” Sabeti said recently. “But the big question is, how is this thing going to be stopped?”
Sabeti and her team made plans to begin reading the genome of the virus as soon as possible. All the drugs, vaccines, and diagnostic tests for Ebola depend critically on the virus’s genetic code. The researchers knew that the code was changing. Could Ebola be evolving away from the defenses against it? Where had it come from? Had it started in one person or had it begun in different people at different times and places? Could Ebola become more contagious, and spread faster?
Sabeti and her team conceived a plan to obtain samples of blood from people infected with Ebola. They would read the genomes of whatever Ebola they could find in the patients’ blood. When monks copied texts by hand in the Middle Ages, they made mistakes. Since Ebola makes errors as it replicates, each genome was like a hand-copied text, and detectable differences would emerge among the genomes; there isn’t just one “strain” of the virus. Ebola is not a thing but a swarm. It is a vast population of particles, different from one another, each particle competing with the others for a chance to get inside a cell and copy itself. The swarm’s genetic code shifts in response to the changing environment. By looking at a few genomes of Ebola, the scientists hoped to grasp an image of the whole virus, which could be conceived of as a life-form visible in four dimensions, as vast amounts of code flowing through time and space. To find the genome, they needed blood.
Teams of epidemiologists and health workers spread out from Kenema and identified twelve more women who were sick with Ebola. All of them had been at the funeral of the faith healer. They were taken to the Kenema hospital and placed in the Lassa ward. Humarr Khan and top officials at the Sierra Leone Ministry of Health were anxious to have the genome of Ebola sequenced, and so Khan and Sabeti, working with the ministry officials, used a method of collecting blood that didn’t interfere with patient care: the researchers scavenged samples of blood serum from tubes left over from clinical care. This material was biohazardous medical waste, intended to be burned in the hospital’s incinerator. “We did everything we could to make no footprint in the way we took samples,” Sabeti said. Blood samples were also taken from thirty-five other people who were suspected of having been exposed to Ebola.
The result was a large number of microtubes of human blood serum collected from forty-nine people. Each microtube was the size of the sharpened end of a pencil and contained a droplet of human blood serum, golden in color and no bigger than a lemon seed. The droplets were mixed with a larger quantity of a sterilizing chemical that kills Ebola. Augustine Goba, the head of the hospital lab, packed the tiny tubes of sterilized blood serum in ice inside a box, then sent the box by DHL Express to Harvard.
Four days later, on June 4th, the box arrived at Sabeti’s lab, where a research scientist named Stephen Gire put on bioprotective gear and carried the box into a tiny biocontainment lab to open it. The samples were supposed to be safe, but Gire was taking no chances. Gire is tall and quiet, and there is an air of precision about him. He is a talented chef, and in 2008 he was offered a chance to compete for a spot on the television show “Top Chef,” but he turned it down and, instead, went to the Democratic Republic of the Congo to set up a lab and study monkeypox, a virus related to smallpox. On Gire’s left forearm is a tattoo showing a particle of monkeypox, a stylish image of the virus’s inner structure that Gire designed himself, and which looks like a nest of crescent moons. Now, in the lab at Harvard with the unopened box of blood samples from Africa, he realized that he had forgotten to bring along a knife. He fished out his car keys, slit open the box, and removed the microtubes. The ice had melted, but the tubes were still cold, and they were visibly safe: the color in the tubes confirmed that the blood serum had been sterilized. Each tube contained around a billion particles of Ebola virus.
An aid worker removes the body of a dead woman in Monrovia, Liberia. Photograph by Kieran Kesner / Rex Features VIA AP
Photograph by Kieran Kesner / Rex Features VIA AP
Gire’s first job was to extract from the blood serum the virus’s genetic material. Gire tested all the samples for the presence of Ebola virus. Of the forty-nine people whose blood samples were in the tubes, fourteen had been infected with Ebola. He could tell just by looking: in those samples, the virus had damaged the blood, and the serum had a murky look, clouded with dead red blood cells. He worked late, spinning all the tubes in a centrifuge and adding chemicals. When he was finished, he had fourteen small, clear droplets of water solution, each in its own tube. In each droplet were vast numbers of broken strands of RNA—shattered fragments of genetic code of the Ebola that had once drifted in the blood of the fourteen people from around Kenema. There were many different genomes in the tubes, for the virus had likely mutated as it multiplied.
The next morning, Gire took a car to the M.I.T. campus, carrying a small box containing the tubes of droplets with the Ebola RNA. There, in a lab at the Broad Institute, he and a colleague named Sarah Winnicki, working alongside two other research teams, prepared the RNA to be decoded. The work took four days, and Gire and Winnicki hardly slept. By the end, they had combined all fourteen samples into a single, crystal-clear droplet of water solution. The drop contained about six trillion snippets of DNA. Each was a mirror image of a piece of RNA from the blood samples. Most of the snippets were human genetic code, but among them were about two hundred billion snippets of code from Ebola. There were also many billions of fragments of code from bacteria and other viruses—from anything that happened to be living in the blood. This droplet was referred to as a library.
“I’m so ready to quit—the pay sucks, and every night I go home reeking of hazelnut.”
Each piece of DNA in the droplet had been tagged with a unique bar code—a short combination of eight letters of DNA code—identifying that particular fragment as having come from one of the fourteen patients. “You could consider each bar-coded fragment of DNA as a kind of book,” Gire said. “The book is bound in covers and has an I.S.B.N. number on it. It’s a short book, so a reader can easily digest it. You can find the book by its I.S.B.N. number, and that’s why the droplet is called a library. The books in the DNA library are bound so that the library can be put in a machine”—a genetic sequencer—“and the machine reads all the books.” The droplet contained many more books of DNA letters than there are books in the Library of Congress. The books were all sitting in one immense, jumbled pile, and what was between their covers was unknown.
On Friday, June 13th, Gire carried a single microtube containing the liquid-droplet library to a logging station in the Genomics Platform of the Broad Institute. The Platform houses a suite of rooms crowded with DNA-sequencing machines. Each machine is a white rectangular box about the size of a chest freezer and costs a million dollars; there are more than fifty of them in the Platform, lined up in rows. Half a dozen technicians tend them around the clock, as they read letters of DNA gathered from biological samples. Recently, the machines have read the genomes of the rabbit, the coelacanth, the malaria parasite, the mosquito that carries malaria, candida fungus, Epstein-Barr virus, and a number of human genes involved in cancer, autism, and schizophrenia.
Using a pipette, a technician sucked up about a tenth of Gire’s Ebola droplet—an amount like a fleck of moisture on a wet day—and placed it on a glass slide known as a flow cell. The fleck of liquid contained the full library of code from the blood of the fourteen Ebola patients. The bit of water spread into channels on the flow cell, which sat in the mouth of an Illumina HiSeq 2500 machine, one of the fastest DNA sequencers in the world.
For the next twenty-four hours, the sequencer worked automatically, pulsing liquids across the flow cell, while lasers shone on it. On the surface of the flow cell, hundreds of millions of fragments of DNA had gathered into hundreds of millions of microscopic colored spots. The colors of the individual spots were changing as the process went on, and a camera took pictures of the changing field of spots and stored the data. Twenty-four hours later, the machine had finished reading Gire’s library of bar-coded fragments of DNA. The data were sent to the Broad Institute’s computer arrays, which assembled all the fragments into finished genetic code—it organized the vast pile of books in the library and placed the letters of all the books in their proper order on shelves. On Sunday, June 15th, Gire and Sabeti got word that the computers had finished their job. The result was twelve full genomes of Ebola virus—the Ebolas that had lived in twelve of the fourteen people. (The computers had not been able to assemble the Ebola genomes from two of the people.) Sabeti and her team started the work of analyzing the code, to see how Ebola was changing.
In early July, Stephen Gire flew to Sierra Leone with another member of Sabeti’s team, and they went to the Kenema hospital, bringing with them lab equipment for use in the Ebola outbreak. Gire was grieved by what he saw. Ebola patients were flowing in from the countryside, dying and terrified. They had filled up the Lassa ward, which had become an Ebola ward, and a second ward—a large white structure with plastic walls and a plastic roof—had been erected. It was full of Ebola patients. The new ward had a plastic viewing window in it, so that people could see and talk to their loved ones inside the ward. Family members of Ebola patients were milling around the window. As Gire recalled, there were shouts of surprise and joy when a patient came to the window and family members saw that the patient was alive and could walk, and cries of sorrow when news came that someone had died. Some in the crowd were silent, baffled by the white building and the moonsuits worn by the health workers. In that part of the world, not everybody believed in the infectious theory of disease, the idea that illnesses can spread through microbes. Why wouldn’t the doctors let people see or touch their loved ones at a funeral? Many people distrusted the government, and spiritual explanations for the disease circulated.
Humarr Khan was working in the Ebola wards. When he came out, and had stripped off his P.P.E., Gire thought that he seemed exhausted and tense. Khan met regularly with international aid workers, and he made countless calls on his cell phone to representatives from the World Health Organization and officials from the Sierra Leone Ministry of Health, pleading for more help, more resources. He called family members—he had nine brothers and sisters, some of whom lived in the United States, and his parents were still alive, in Lungi, a town not far from Freetown, the capital. He spoke with Pardis Sabeti; he planned to join her group at Harvard in a few months. He was fascinated by genomics and he wanted to know how the sequencing of Ebola was going. He couldn’t stand the bureaucracy of the outbreak, Sabeti told me, and he would return to the Ebola wards as if they were a refuge from trouble. He seemed more at ease wearing P.P.E. and caring for patients.
Khan had been running the Lassa program for almost a decade. In 2004, his predecessor, Aniru Conteh, accidentally pricked himself with a needle contaminated with blood from a pregnant woman who had Lassa. Conteh died twelve days later, of Lassa fever, tended by his own nurses. For months, the government couldn’t find any doctor willing to run the Lassa program. Khan, who had just finished his internship at the Sierra Leone College of Medicine, agreed to take the job.
Khan arrived driving a battered old car. He was thirty, a modest, handsome man who smiled and joked playfully with people. Khan took up his work and gave patients exceptional attention. One day, a U.S. graduate student named Joseph Fair fell desperately ill with bloody diarrhea. Khan paid a visit to Fair at his room in a nearby Catholic mission, and that was when Fair discovered that Khan had a beautiful bedside manner. After prescribing antibiotics, Khan jovially said to him, “You’ll be fine.” But, leaving the room, Khan forgot to close the door. Moments later, Fair heard him blurt out to somebody, “This guy is dying! I can’t have an expat die on me!” Fair got better, and he and Khan soon became friends. A few years later, they were having a beer in a bar in New Orleans when Fair told Khan that the first time they met he had heard Khan say he was dying. “Well, you were dying,” Khan answered. Fair said, “You didn’t tell me.” Khan burst out laughing. “I would say you were dying? You were my patient. Can you imagine?”
Khan worked long hours in the Ebola wards, trying to reassure patients. Then one of the nurses got sick with Ebola and died. She hadn’t even been working in the Ebola ward. The virus particles were invisible, and there were astronomical numbers of them in the wards; they were all over the floor and all over the patients.
There are two distinct ways a virus can travel in the air. In what’s known as droplet infection, the virus can travel inside droplets of fluid released into the air when, for example, a person coughs. The droplets travel only a few feet and soon fall to the ground. The other way a virus can go into the air is through what is called airborne transmission. In this mode, the virus is carried aloft in tiny droplets that dry out, leaving dust motes, which can float long distances, can remain infective for hours or days, and can be inhaled into the lungs. Particles of measles virus can do this, and have been observed to travel half the length of an enclosed football stadium. Ebola may well be able to infect people through droplets, but there’s no evidence that it infects people by drying out or getting into the lungs on dust particles. In 1989, a virus known today as Reston, which is a filovirus related to Ebola, erupted in a building full of monkeys in Reston, Virginia, and travelled from cage to cage. One possible way, never proved, is that the virus particles hitched rides in mist driven into the air by high-pressure spray hoses used to clean the cages, and then circulated in the building’s air system. A rule of thumb among Ebola experts is that, if you are not wearing biohazard gear, you should stand at least six feet away from an Ebola patient, as a precaution against flying droplets.
Some patients with Ebola become disoriented, struggle and thrash, and fall out of bed. They can get a bloody nose, which makes them sneeze. They can have projectile vomiting, and they can cough while they are vomiting. Some become incontinent, and all the fluids that come out of their bodies are increasingly saturated with Ebola particles. The new plastic-walled Ebola ward at Kenema had a type of bed in it, common in African hospitals, known as a cholera bed. A patient with cholera suffers from uncontrollable watery diarrhea. A cholera bed has a plastic-covered mattress with a hole in the center. A bucket is placed on the floor under the hole and the patient defecates through it into the bucket. In the Ebola ward, the nurses were emptying the buckets and trying to keep things clean, but it was impossible. Then some of the nurses began skipping work. In the tropical heat, the smell of the Ebola wards became intense.
Around July 12th, Joseph Fair, who had been working with the World Health Organization in Freetown, two hundred miles away, travelled to Kenema, a drive of several hours, and went looking for his friend Dr. Khan. Fair found him but couldn’t speak with him, he told me later. Khan was inside the plastic Ebola ward, and the place was a mess. There were thirty or more Ebola patients in the ward, lying on cholera beds, and the floor was splashed with everything that can come out of the human body. Khan was making rounds, with one nurse, both of them wearing P.P.E.
Daniel Bausch, an American Ebola doctor who had been helping at Kenema, and his colleagues recently wrote that Khan had remarked, “I am afraid for my life, I must say. . . . Health workers are prone to the disease, because we are the first port of call for somebody who is sickened.” They also quoted Khan’s sister Isatta as saying, “I told him not to go in there, but he said, ‘If I refuse to treat them, who would treat me?’ ” Perhaps Khan was thinking of his predecessor Dr. Conteh, dying in his own ward.
Alex Moigboi, a popular man who had worked in the hospital for many years, came down with Ebola. Then the head nurse, Mbalu Fonnie, a widow who sometimes used the last name Sankoh, and who had worked at the hospital since it opened, in the nineteen-nineties, began feeling weak and shivery and ran a fever. At first, she downplayed her symptoms and continued working seven days a week, fourteen to sixteen hours a day. She hoped that she had malaria, and gave herself an I.V. drip of malaria medicine, but she didn’t get better. She tested positive for Ebola. That same day, two other Kenema nurses, Fatima Kamara and Veronica Tucker, also tested positive for Ebola. Moigboi died on July 19th, and Fonnie died two days later.
Many of the staff at Kenema became terrified and began staying home from work. Khan ended up working in the Ebola wards with little or no support. Sierra Leone’s medical-care system, sparse and rudimentary to begin with, was collapsing under the strain of Ebola, and the international aid groups that worked in Ebola outbreaks were stretched thin. Doctors Without Borders was coping with Ebola patients in a treatment center at Kailahun, in eastern Sierra Leone, fifty miles from Kenema. In Liberia, doctors and nurses with Samaritan’s Purse, a Christian organization, were overrun with patients at a hospital called ELWA, near Monrovia. Khan talked regularly with Pardis Sabeti. “We are all alone here,” he said to her one day. She told him that she and her colleagues in the War Room were rushing people and equipment to him, and they were calling around the world, looking for more doctors and more help. “People and help were coming,” Sabeti told me later, “but it was nowhere near enough.”
Sabeti warned Khan about stress and overwork. “The most important thing is your safety. Please take care of yourself.”
He told her, “I have to do everything I can to help these people,” and then he would put on his gear and go back into the Ebola wards. Khan was a general in a battle where many of his troops were dead or fleeing.
On July 19th, at a large staff meeting, people noticed that Khan didn’t look well. The next day, he didn’t come to work. He had isolated himself at home. The following morning, he requested a test. One of the lab technicians went to his house to draw blood: it was positive for Ebola. Khan didn’t want to be treated at Kenema, because he didn’t want his staff to see him develop symptoms, and he felt that his presence would further demoralize them. The next day, he climbed into an ambulance, which carried him along rutted dirt roads to the Ebola ward in Kailahun.
At the treatment center in Kailahun, there was a freezer powered by a generator, and inside the freezer were three small plastic bottles containing a frozen water solution. In it were antibodies, Y-shaped molecules that are produced naturally by the immune systems of mammals as a defense against invading microbes. The liquid was ZMapp, an experimental drug for the treatment of Ebola, and the three bottles amounted to what might be a course of ZMapp for one human being. The drug was untested in humans. During the previous decade, a group of scientists, working with very little money and virtually no encouragement from the community of Ebola experts, had developed the drug. The effort involved dozens of people, but the principal researchers were Larry Zeitlin, the president of Mapp Biopharmaceutical, a biotech company in San Diego; Gene Garrard Olinger, a contractor with the National Institute of Allergy and Infectious Diseases division of the National Institutes of Health; and Xiangguo Qiu and Gary Kobinger, researchers at the Public Health Agency of Canada’s research facility in Winnipeg. ZMapp was a cocktail of three antibodies that seemed especially potent in killing Ebola. Mapp Biopharmaceutical and the manufacturer, Kentucky BioProcessing, had developed a method of growing it in tobacco plants.
In April of 2014, three months before Khan fell ill, Kobinger and his group in Canada tested ZMapp for the first time in monkeys infected with Ebola. They gave the monkeys a thousand times the lethal dose of Ebola. To the researchers’ surprise, the drug saved the monkeys. ZMapp could work even when the animal seemed close to death. Kobinger and his team found that they had to give the animal three doses of ZMapp spaced a few days apart. Kobinger compared this to three punches from a prizefighter: the first two punches knocked Ebola down and the third ended the fight. In late June, while Ebola was starting to blow up across West Africa, Kobinger travelled from his lab in Winnipeg to Kailahun with lab equipment for the doctors there, along with the three plastic bottles of ZMapp, and left the bottles in the Kailahun freezer. He wanted to see how ZMapp held up in the tropical climate, where the heat and an uncertain electricity supply can ruin a drug’s effectiveness. He had no idea that it would be used.
The government of Sierra Leone regarded Humarr Khan’s plight as a national crisis. As soon as Khan became ill, a government official sent out an e-mail to Ebola experts around the world, asking for information about any drug or vaccine that might help him. In a series of international conference calls, officials from the World Health Organization, the U.S. Centers for Disease Control and Prevention, the government of Sierra Leone, the Public Health Agency of Canada, scientists from the United States Army, and health workers from Doctors Without Borders, which was running the Kailahun Ebola center, debated how to treat Khan. Many of the people on the phone knew him, and this was a matter of life and death.
The debate quickly centered on ZMapp, which seemed to show more promise than other drugs. Why should Khan, and not other patients, get any experimental drug? What if he died? ZMapp had been tested in some monkeys a few months earlier, but what was the significance of that? It was made from mouse-human antibodies that had been grown in tobacco plants. If such substances enter the bloodstream, a person might have a severe allergic reaction. If something went wrong with the drug, there was no intensive-care unit in Kailahun. The population of Sierra Leone would be furious if the West was seen to have killed Khan, an African scientist and a national hero, with an experimental drug. But if he wasn’t given the ZMapp, and he died, people might say that the West had withheld a miracle drug from him. “I was making sure my tone of voice stayed neutral,” Kobinger recalled. The debate and the calls went on for three days.
Meanwhile, at the ELWA hospital, two hundred miles to the south, a fifty-nine-year-old American health worker named Nancy Writebol got a fever. She tested positive for malaria and went to bed in her house, on the grounds of the hospital, where she lived with her husband, David Writebol. Soon afterward, Kent Brantly, a thirty-three-year-old American doctor with Samaritan’s Purse at ELWA, called the medical director of disaster response for Samaritan’s Purse, Lance Plyler. “Don’t freak out, Lance, but I think I’ve got a fever,” Brantly said. He put himself into isolation in his house on the hospital grounds, and Samaritan’s Purse sent a sample of his blood to the National Reference Laboratory of Liberia. Plyler told me that he didn’t want anybody to know that one of his doctors might have Ebola, so he labelled the tube with a fictitious name, Tamba Snell.
The National Reference Lab of Liberia is a former chimpanzee-research center and sits at the end of a dirt road in the forest near Monrovia’s international airport. It is well staffed and well equipped. An American virologist named Lisa Hensley had been working there with Liberian and American colleagues, testing dozens of clinical samples of liquids from the bodies of people suspected of having Ebola. Hensley works with the National Institute of Allergy and Infectious Diseases, and has been doing research on Ebola in U.S. government biocontainment labs for more than fifteen years. She and her colleagues, wearing pressurized P.P.E. suits, were using devices called PCR machines to find out if Ebola was present in the samples, in order to help doctors in Liberia identify people who were infected. Technicians at the lab tested the blood of Tamba Snell. It came up negative for Ebola, and Hensley e-mailed the result to a doctor at Samaritan’s Purse. The real Tamba Snell, Kent Brantly, got sicker.
On July 25th, the international groups finally came to a decision about Humarr Khan. ZMapp was too risky and would not be given to him. Khan was informed; it is not clear that he was brought in to the decision. That same day, his brother Sahid, in Philadelphia, began frantically calling Kailahun in an effort to speak to him. Sahid had been calling Humarr’s cell phone for days but had got no answer. Sahid got somebody at Kailahun on the phone and demanded to speak with his brother. “It is not possible to speak to Humarr,” he was told. Sahid blew up. “Then I want a picture of him to prove he is still alive!” he shouted. Soon afterward, somebody texted him a photo of his brother. In the image, Humarr is sitting on a plastic chair, slumped, and his eyes are heavy-lidded. He appears to be exhausted and turned inward, though a slight smile flickers on his face. Sahid believes that the smile was for the sake of their mother, an attempt to tell her not to worry.
At the lab in Monrovia, Lisa Hensley and her group received another sample from Tamba Snell. Shortly afterward, Hensley got an e-mail from an official with the C.D.C. saying that the blood came from “one of our own.” Hensley understood this to mean that an outbreak responder might have Ebola. Then another sample came in, with the name Nancy Johnson. Hensley knew that the names were fictitious. The lab wasn’t staffed that day—it was July 26th, Liberian Independence Day, a national holiday. Nevertheless, Hensley and a colleague, Randal Schoepp, put on P.P.E. and went into the lab. They began with the blood of Tamba Snell. The machines worked fast: he had Ebola. Hensley e-mailed Lance Plyler: “I am very sorry to inform you that Tamba Snell is positive.” Later that day, she texted him: Nancy Johnson had Ebola, too.
At elwa, Plyler went to the house where Kent Brantly was isolated, in bed, and was distressed to see how ill he looked. “I hate to tell you that you have Ebola,” he said. After a moment, Brantly said, “I really did not want you to say that.” Plyler immediately decided that he would do all he could. He knew that there were experimental drugs for Ebola. Doctors from Samaritan’s Purse sent an e-mail to a C.D.C. official who was stationed in Monrovia: they wanted to talk to a researcher with direct experience in the development of the drugs. They wanted that person to put Plyler in touch with anyone who might have access to these possible therapies.
That person turned out to be Lisa Hensley, the scientist in Monrovia who had just tested Brantly’s and Writebol’s blood. She sent information to Samaritan’s Purse and offered to visit ELWA as soon as possible. She couldn’t get out until the following evening, and the roads weren’t entirely safe after dark. The hospitals in Monrovia were full of Ebola patients, and the medical system was crumbling. In the countryside, medical-outreach teams had been attacked by mobs of frightened people. Hensley called the U.S. Embassy in Monrovia and arranged for an Embassy car and driver to take her to ELWA. She arrived at ten o’clock that night; Plyler was waiting in his car. They drove through the compound until they came to a small house, painted white, where a lighted window was opened just a crack. Kent Brantly was sitting in bed behind the window, with his laptop. He was researching his case, and he told Hensley that he knew about antibodies to Ebola.
Hensley had done laboratory research on experimental drugs and vaccines for Ebola. Speaking to Brantly through the window, she summarized nineteen possible options. Almost none of them had been tested in humans. In January, Tekmira Pharmaceuticals had begun testing a drug, TKM-Ebola, in humans, evaluating it for safety. It had shown decent results in monkeys, but the drug had been put on partial hold while the company collected more information for the Food and Drug Administration. There was a drug called T705, which had been tested in Japan, in humans, against influenza virus, and it might have some effect on Ebola. Hensley told Brantly that she had participated in a study of a drug called rNAPc2, an anticoagulant made by a company called Nuvelo; the drug saved one of three monkeys it was tested on. Brantly focussed his attention on ZMapp. It had saved monkeys even when they were deep into the illness, as he was now. But, still, he didn’t know. When Hensley finished, Brantly’s voice came out through the window: “What would you do, Lisa?”
She couldn’t tell him what to do. “These are all very personal decisions,” she said. Then she told him that she had been exposed to Ebola, sixteen years earlier. At the age of twenty-six, working in a spacesuit with liquids full of Ebola particles, she had cut her finger with scissors, which had gone through two layers of gloves. The only experimental treatment at that time was a horse serum made by the Russians; this could kill her, and she had decided not to use it unless she was certain that she had contracted Ebola. On the night of the accident, after a meeting to analyze what had happened, she was sent home to her apartment. She called her parents and told them that she might come down with Ebola and that they would have to collect her belongings and take her cat home with them.
Brantly listened, and said that he probably would choose ZMapp for himself, based on the data, even though it had never been tested on humans. Hensley offered to donate blood if he had hemorrhages. Plyler then drove her across the compound to Nancy Writebol’s house. Writebol was asleep close to a window. Her husband and a nurse both put on P.P.E. and woke her, and Hensley spoke with her from outside the house. Meanwhile, Hensley noticed that the window was wide open, and Writebol began coughing. A ceiling fan blew gusts of air out the window and across Hensley and the others. Hensley could smell the air from the bedroom. She took a step back but didn’t say anything. Later that night, in her hotel room, Hensley sent a text to Lance Plyler. “You guys make me a little bit nervous,” she typed, and she advised them to wear breathing masks outside the windows of the two patients.
On July 28th, Gary Kobinger, of the Public Health Agency of Canada, received an e-mail from Lance Plyler asking for ZMapp to be sent to ELWA as quickly as possible. Kobinger told him that the nearest course of the drug was sitting in a freezer in Kailahun, in Sierra Leone, across an international border. By now, Humarr Khan was close to death. Hensley had not taken part in the debate over whether to give ZMapp to Khan, but she knew about the decision.
The drug would have to be flown from Kailahun, but there was no airfield there; the nearest was in a town called Foya. A few days earlier, a team from the Sierra Leone Ministry of Health had been attacked in Foya, and a ministry vehicle was burned; residents were fleeing the area. The U.S. Embassy in Monrovia asked Lisa Hensley to pick up the drug and arranged a helicopter for her.
“You don’t whisper anymore.”
The chopper was an old gray Russian Mi-8, flown by two Ukrainian pilots. A colonel in the U.S. Marine Corps accompanied her—to provide peace of mind, he told her. A heavy rain was falling, and Hensley and the colonel sat in the helicopter for hours on the tarmac. During those hours, in Kailahun, Humarr Khan died. Finally, during a break in the weather, the helicopter took off and headed north. Hensley, wearing ear protectors, sat buckled on a bench facing the colonel. She could see almost nothing out the window except moisture whipping across the glass, but now and then she caught a glimpse of a ridge covered in jungle slipping by below. She grew anxious, especially when the colonel remarked, “We’ve been flying in periods of zero visibility.”
In this outbreak, everybody was flying in near-zero visibility. Below the helicopter, lost in the rain, Ebola was maneuvering in secret. No drugs or vaccines were known to work against it in people; Hensley was on her way to get one sample of one experimental compound. Later, she told me, “If you are walking by a lake and somebody is drowning, you can’t not try to save them. People are drowning in Ebola.”
She was a single mother, with a nine-year-old son she’d left back in Maryland, in the care of her parents. “If we don’t help, what message are we sending to our children?” she said to me one day. “Our children are going to inherit these problems, and people are dying. Part of the responsibility of a parent is to teach our children how to be responsible. We have to set the example for our staff, our families, and the patients in Africa.”
Hensley dozed off, and when the chopper touched down in Foya she discovered that a plane from Samaritan’s Purse had already left with the drug. The helicopter flew back to Liberia.
At the ELWA hospital, Lance Plyler, with the drug now in his hands, agonized about whether he should give it to Writebol or to Brantly. He found some words in the Book of Esther: “Who knows whether you have come to the kingdom for such a time as this?” Writebol was extremely ill by now, but he found Brantly in surprisingly good condition, working on his laptop in bed. Brantly was more concerned about Writebol. “Give the drug to Nancy—I’ll be getting out of here in a couple of days,” he told Plyler. An evacuation jet had been ordered, and he was evidently thinking of that. Still, Plyler put off the decision. Another night passed.
On the morning of July 31st, Plyler went to see Nancy Writebol, and decided to give her the drug. She seemed close to the end stage of Ebola-virus disease; she had developed a sea of red spots and papules across her torso—signs of hemorrhages under the skin—and she was beginning to bleed internally. She could crash at any time: lose blood pressure, go into shock, and die. One of the bottles was taken out of the freezer, and Plyler had Writebol hold it in her armpit to defrost it.
Around seven o’clock that evening, Plyler went to Brantly’s house to see how he was doing. When he looked in the window, he was stunned. Brantly had abruptly gone into the end-stage decline. His eyes were sunken, his face was a gray mask, and he was breathing in irregular gasps. “A clinician knows the look,” Plyler told me later. “He was dying.” Brantly, a clinician himself, realized that he was on the verge of a breathing arrest. With no ventilators at the hospital, he wouldn’t make it through the night.
Plyler made a decision. “Kent, I’m going to give you the antibodies.” He would split the three doses, giving one bottle to Brantly, the second bottle to Writebol, and the third bottle to whichever of them was not evacuated.
A nurse got the bottle from under Writebol’s arm. Writebol said that she was glad for Brantly to have it. While Plyler watched, a doctor named Linda Mabula suited up and went into Brantly’s house, where she prepared an I.V. drip. The plan was to drip the first dose into him very slowly, so that the antibodies wouldn’t send him into shock. Plyler stayed by the window and prayed with Brantly. After less than an hour, Brantly began to shake violently, a condition called rigors. It occurs in people who are near death from an overwhelming bacterial infection. Plyler had a different feeling about these rigors. “That’s just the antibodies kicking the virus’s butt,” he told Brantly through the window.
Three hours later, Lisa Hensley got a text from Lance Plyler: “Kent is about halfway into the first dose. Honestly he looks distinctly better already. Is that possible?” Hensley texted back to say that monkeys on the brink of death had shown improvement within hours. Two days later, having received one dose of ZMapp out of the required three doses, and a blood transfusion from a fourteen-year-old boy who had recovered from Ebola, Kent Brantly walked onto the evacuation plane. At Emory University Hospital, in Atlanta, he received two more doses of ZMapp, which had been sent from the tobacco facility in Kentucky, and was discharged from the hospital after two weeks, free of the virus.
Nancy Writebol had a different experience. She did not improve noticeably when she got the first dose of ZMapp, and she developed intense itching in her hands, which seemed to be an allergic reaction to the drug. She continued to have internal hemorrhages afterward, and was given a blood transfusion to make up what she was losing. Nevertheless, she survived. She was evacuated to Emory University Hospital two days later and received more ZMapp and another blood transfusion there.
As of this writing, the world’s supply of ZMapp is temporarily exhausted. It was given to five more patients with Ebola, including a Spanish priest, who died shortly after getting the first dose. More of the drug is growing in tobacco plants in a building in Kentucky. The plants have enough of the drug in them to make twenty to eighty treatment courses of ZMapp in the next two months, as long as there are no glitches in the process. The U.S. government and Mapp Biopharmaceutical are scrambling to get more plants growing, to increase production, but the scale-up will not be easy. The drug remains untested, and nobody can say whether it will ever become a weapon in the Ebola wars.
At two o’clock in the afternoon on July 31st, the funeral of Humarr Khan began in Kenema. It was attended by five hundred people, including townspeople, scientists, health workers, and Sierra Leone government ministers. Many wept uncontrollably. The gravediggers encountered rocks, and it took them hours to dig deep. At ten o’clock that night, in the moments when Kent Brantly was shaking with rigors as ZMapp flowed into his body, the gravediggers finished burying the body of Khan at the Kenema hospital.
As Khan lay dying, Pardis Sabeti composed a song for him and the other Kenema workers, called “One Truth.” It had the line “I’m in this fight with you always.” She had hoped that some day she could sing it to him, but by then he was already in isolation. When she received the news of his death, she was “absolutely devastated,” she said. “I can’t even begin to describe the feeling of loss for the world.” Equally devastating were the deaths of the staff members who had stayed to work in the wards at Kenema.
Through the summer, Sabeti and her group continued to read the Ebola genomes. They published them in real time, on the Web site of the National Center for Biotechnology Information, so that scientists anywhere could see the results immediately. Then, in late August, they published a paper in Science detailing their results. They had sequenced the RNA code of the Ebolas that lived in the blood of seventy-eight people in and around Kenema during three weeks in May and June, just as the virus was first starting chains of infection in Sierra Leone. The team had run vast amounts of code through the sequencers, and had come up with around two hundred thousand individual snapshots of the virus, in the blood of the seventy-eight people, and had watched it mutate over time. They could see who had given the virus to whom. They could see exactly how it had mutated each time it grew in one person and jumped to the next. The snapshots, taken together, amounted to a short video of Ebola. You could imagine the virus as a school of fish, with each particle of Ebola a fish. The fish were swimming, and as they swam and multiplied they changed, until the school had many kinds of fish in it and was growing exponentially in size, with some kinds of fish better at swimming than others.
“Never get a tattoo when you’re drunk and hungry.”
Gire and Sabeti’s group also found that the virus had started in one person. It could have been the little boy in Meliandou, but there is no way to tell for sure right now. After that, the swarm mutated steadily, its code shifting as it palpated the human population. As the virus jumped from person to person, about half the time it had a mutation in it, which caused one of the proteins in the virus to be slightly different. By the time the virus reached Sierra Leone, travelling in the bodies of the women who had attended the funeral of the faith healer, it had become two genetically distinct swarms. Both lineages of the virus moved from the funeral into Sierra Leone. Already, some of the mutations were making Ebola less visible to the tests for it.
“It shows that you can analyze Ebola in real time,” Sabeti said. “This virus is not a single entity. Now we have an entry into what the virus is doing, and now we can recognize what we are battling with at every point in time.”
The Science paper included five authors who died of Ebola, including Humarr Khan, the head nurse Mbalu Fonnie, and the nurse Alex Moigboi. “There are lifetimes in that paper,” Sabeti said. A thousand more vials of human blood with Ebola in them are sitting in freezers in Kenema waiting for bureaucratic clearance so that they can be flown to Harvard and sequenced in the machines, and scientists can see what the swarm has been doing more recently.
The question often asked is whether Ebola could evolve to spread through the air in dried particles, entering the body along a pathway into the lungs. Eric Lander, the head of the Broad Institute, thinks that this is the wrong question to ask. Lander is tall, with a square face and a mustache, and he speaks rapidly and with conviction. “That’s like asking the question ‘Can zebras become airborne,’ ” he said. In order to become fully airborne, Ebola virus particles would need to be able to survive in a dehydrated state on tiny dust motes that remain suspended in the air and then be able to penetrate cells in the lining of the lungs. Lander thinks that Ebola is very unlikely to develop these abilities. “That would be like saying that a virus that has evolved to have a certain life style, spreading through direct contact, can evolve all of a sudden to have a totally different life style, spreading in dried form through the air. A better question would be ‘Can zebras learn to run faster?’ ”
There are many ways by which Ebola could become more contagious even without becoming airborne, Lander said. For example, it could become less virulent in humans, causing a milder disease and killing maybe twenty per cent of its victims instead of fifty per cent. This could leave more of them sick rather than dead, and perhaps sick for longer. That might be good for Ebola, since the host would live longer and could start even more chains of infection.
In the lab in Liberia, Lisa Hensley and her colleagues had noticed something eerie in some of the blood samples they were testing. In those samples, Ebola particles were growing to a concentration much greater than had been seen in samples of human blood from previous outbreaks. Some blood samples seemed to be supercharged with Ebola. This, too, would benefit the virus, by enhancing its odds of reaching the next victim.
“Is it getting better at replicating as it goes from person to person?” Hensley said. She isn’t at all sure; maybe in previous outbreaks some people had had these profusions of particles in their blood. “We have to go back to the lab to answer this question.”
Sun Tzu, the great Chinese strategist, wrote that one of the rules of war is to know the enemy. Sabeti and her team now had a way to watch Ebola as it changed; they had the enemy in sight. This meant that the tests for Ebola could be updated quickly as the virus changed, and that the scientists might also be able to see it mutating in some dangerous direction.
Meanwhile, scientists have been developing weapons against the virus and are starting to test them. The scientists who came up with ZMapp, along with Kentucky BioProcessing, were racing to increase the production of ZMapp and to get it tested as a new drug in patients infected with Ebola. The hope is to get the drug through clinical trials and gain the support of a regulatory agency. Even at increased production speed, the supply of ZMapp would still be nowhere near enough to treat the population, but it might be enough—provided it was effective—to kill Ebola in some infected people. If there were a drug that could save somebody from Ebola, this might help encourage health professionals to work in Ebola wards, knowing that there would be a treatment for them if they got infected.
In addition to many drug candidates, there are vaccines in development. In early September, the National Institutes of Health began testing a vaccine, made by a division of GlaxoSmithKline and based on an adenovirus, on twenty volunteers. Another vaccine, called VSV-EBOV, developed by the Public Health Agency of Canada and licensed to NewLink Genetics, started human trials last week. It seems possible that some time next year a vaccine may be available for use on people who have already been exposed to Ebola, though it will still not be cleared for general use. If a vaccine is safe and shows effectiveness against Ebola, and if it can be transported in the tropical climate without breaking down, then vaccinations against Ebola could someday begin.
If a vaccine works, then the vaccinators might conceivably set up what’s known as ring vaccinations around Ebola hot spots. In this technique, medical workers simply vaccinate everybody in a ring, miles deep, around a focus of a virus. It works like a fire break; it keeps the fire from spreading. Ring vaccination was the key to wiping out the smallpox virus, which was declared eradicated in 1979, but whether the ring technique—provided there was a good vaccine—would work against Ebola nobody can say. In any case, epidemiologists would not give up trying to trace cases in order to break the chains of infection.
In the U.S. and Europe, hospitals have made fatal mistakes in protocol as they engage with Ebola for the first time—errors that no well-trained health worker in Africa would likely make. But they will learn. By now, the warriors against Ebola understand that they face a long struggle against a formidable enemy. Many of their weapons will fail, but some will begin to work. The human species carries certain advantages in this fight and has things going for it that Ebola does not. These include self-awareness, the ability to work in teams, and the willingness to sacrifice, traits that have served us well during our expansion into our environment. If Ebola can change, we can change, too, and maybe faster than Ebola. ♦
Sign up for our daily newsletter: the best of The New Yorker every day.
Go

'''

print( calcPairSim([doc_0, doc_1]) )
