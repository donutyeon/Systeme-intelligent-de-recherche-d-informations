import nltk 
import math
import numpy as np

def prep():
        f1= open('D1.txt', 'r')
        f2= open('D2.txt', 'r')
        f3= open('D3.txt', 'r')
        f4= open('D4.txt', 'r')
        text1=f1.read()
        text2=f2.read()
        text3=f3.read()
        text4=f4.read()

        ExpReg2 = nltk.RegexpTokenizer('(?:\w\.)+|\d+(?:\.\d+)?%?|(?:\w|[`-]\w)+|\w+')
        text2=text2.replace(",","").replace("(","").replace(")","")
        Termes2 = ExpReg2.tokenize(text2)

        text1=text1.replace(",","").replace("(","").replace(")","")
        Termes1 = ExpReg2.tokenize(text1)

        text3=text3.replace(",","").replace("(","").replace(")","").replace('"','')
        Termes3 = ExpReg2.tokenize(text3)

        text4=text4.replace(",","").replace("(","").replace(")","").replace('"','')
        Termes4 = ExpReg2.tokenize(text4)

        MotsVides = nltk.corpus.stopwords.words('english')
        TermesSansMotsVides1 = [terme for terme in Termes1 if terme.lower() not in MotsVides]
        TermesSansMotsVides2 = [terme for terme in Termes2 if terme.lower() not in MotsVides]
        TermesSansMotsVides3 = [terme for terme in Termes3 if terme.lower() not in MotsVides]
        TermesSansMotsVides4 = [terme for terme in Termes4 if terme.lower() not in MotsVides]

        global Porter
        Porter = nltk.PorterStemmer()
        TermesNormalisation1 = [Porter.stem(terme) for terme in TermesSansMotsVides1]
        TermesNormalisation2 = [Porter.stem(terme) for terme in TermesSansMotsVides2]
        TermesNormalisation3 = [Porter.stem(terme) for terme in TermesSansMotsVides3]
        TermesNormalisation4 = [Porter.stem(terme) for terme in TermesSansMotsVides4]

        TermesFrequence1 = {}
        for terme in TermesNormalisation1:
                if (terme in TermesFrequence1.keys()):
                        TermesFrequence1[terme] += 1
                else:
                        TermesFrequence1[terme] = 1

        TermesFrequence2 = {}
        for terme in TermesNormalisation2:
                if (terme in TermesFrequence2.keys()):
                        TermesFrequence2[terme] += 1
                else:
                        TermesFrequence2[terme] = 1

        TermesFrequence3 = {}
        for terme in TermesNormalisation3:
                if (terme in TermesFrequence3.keys()):
                        TermesFrequence3[terme] += 1
                else:
                        TermesFrequence3[terme] = 1

        TermesFrequence4 = {}
        for terme in TermesNormalisation4:
                if (terme in TermesFrequence4.keys()):
                        TermesFrequence4[terme] += 1
                else:
                        TermesFrequence4[terme] = 1
        global TermesFrequence
        TermesFrequence={}
        TousTermes=TermesNormalisation1+TermesNormalisation2+TermesNormalisation3+TermesNormalisation4
        for terme in TermesNormalisation1:
                if ((terme,1) in TermesFrequence.keys()):
                        TermesFrequence[(terme,1)] += 1
                else:
                        TermesFrequence[(terme,1)] = 1
        for terme in TousTermes:
                if terme not in TermesNormalisation1:
                        TermesFrequence[(terme,1)] = 0

        for terme in TermesNormalisation2:
                if ((terme,2) in TermesFrequence.keys()):
                        TermesFrequence[(terme,2)] += 1
        else:
                TermesFrequence[(terme,2)] = 1
        for terme in TousTermes:
                if terme not in TermesNormalisation2:
                        TermesFrequence[(terme,2)] = 0

        for terme in TermesNormalisation3:
                if ((terme,3) in TermesFrequence.keys()):
                        TermesFrequence[(terme,3)] += 1
                else:
                        TermesFrequence[(terme,3)] = 1
        for terme in TousTermes:
                if terme not in TermesNormalisation3:
                        TermesFrequence[(terme,3)] = 0

        for terme in TermesNormalisation4:
                if ((terme,4) in TermesFrequence.keys()):
                        TermesFrequence[(terme,4)] += 1
                else:
                        TermesFrequence[(terme,4)] = 1
        for terme in TousTermes:
                if terme not in TermesNormalisation4:
                        TermesFrequence[(terme,4)] = 0

def freq1(dico, docu):
        keys = dico.keys()
        docu_keys = []
        for each in keys:
                if docu == each[1]:
                        docu_keys.append(each)
        response = {}
        for key in docu_keys:
                response[key[0]] = dico[key]
        return response

def freq2(dico, terme):
        terme = terme.strip()
        terme = Porter.stem(terme.lower())
        keys = dico.keys()
        docu_keys = []
        for each in keys:
                if terme == each[0]:
                        docu_keys.append(each)
        response = {}
        for key in docu_keys:
                response[key[1]] = dico[key]
        return response

def poid(dico, terme, docu):
        freqs = freq2(dico,terme)
        try:
                term_freq_in_docu = freqs[docu]
                values = []
                for each in dico.keys():
                        if each[1] == docu:
                                values.append(dico[each])
                max_val = max(values)
                c = 0
                for each in freqs.keys():
                        if freqs[each] !=0:
                                c+=1
                log_part = math.log10((len(freqs)/c)+1)
                other_part = term_freq_in_docu/max_val
                return other_part*log_part
        except: pass

def poid_normalise(dico, terme, docu):
        try:
                k = 2
                b = 0.5
                distinct_docus = {}
                for each in dico.keys():
                        if each[1] in distinct_docus.keys():
                                distinct_docus[each[1]] += dico[each]
                        else: distinct_docus[each[1]] = dico[each]
                
                freqs = freq2(dico,terme)
                term_freq_in_docu = freqs[docu]
                c = 0
                for each in freqs.keys():
                        if freqs[each] !=0:
                                c+=1
                values = []
                for each in distinct_docus.keys():
                        values.append(distinct_docus[each])
                part_idk = k*((1-b)+b*(distinct_docus[docu]/np.mean(values)))
                log_part = math.log10((len(freqs)-c+0.5)/(c+0.5))
                return ((term_freq_in_docu/(part_idk+term_freq_in_docu))*log_part)
        except: pass

def get_weights(dico, terme):
        frequencies = freq2(dico, terme)
        new_dict = {}
        for each in frequencies.keys():
                poid_simple = poid(dico, terme, each)
                poid_normal = poid_normalise(dico, terme, each)
                new_dict[each] = (frequencies[each], poid_simple, poid_normal)
        return new_dict

def make_struct_dico(dico):
        dict_copy = {}
        for each in dico.keys():
                poid_simple = poid(dico, each[0], (each[1]))
                poid_normal = poid_normalise(dico,each[0], each[1])
                dict_copy[each] = (dico[each], poid_simple, poid_normal)
        return dict_copy
