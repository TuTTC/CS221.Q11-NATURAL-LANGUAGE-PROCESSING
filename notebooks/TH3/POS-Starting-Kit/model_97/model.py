import pickle
import numpy as np
from collections import defaultdict
import re


class POSTagger:
    def __init__(self):
        self.tag_counts = defaultdict(int)
        self.tag_tag_counts = defaultdict(lambda: defaultdict(int))
        self.tag_tag_tag_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # Trigram
        self.tag_word_counts = defaultdict(lambda: defaultdict(int))
        self.suffix_tag_counts = defaultdict(lambda: defaultdict(int))
        self.prefix_tag_counts = defaultdict(lambda: defaultdict(int))
        self.word_tag_counts = defaultdict(lambda: defaultdict(int))
        self.shape_tag_counts = defaultdict(lambda: defaultdict(int))  # Word shape
        self.tags = []
        self.vocab = set()
        self.total_words = 0
        self.tag_to_idx = {}
        self.idx_to_tag = {}
        
    def _get_word_shape(self, word):
        """Get word shape pattern"""
        shape = []
        for c in word:
            if c.isupper():
                shape.append('X')
            elif c.islower():
                shape.append('x')
            elif c.isdigit():
                shape.append('d')
            else:
                shape.append(c)
        # Collapse consecutive same characters
        result = []
        prev = None
        for s in shape:
            if s != prev:
                result.append(s)
                prev = s
        return ''.join(result)
        
    def fit(self, Sents, POSs):
        """Train improved HMM model with rich features"""
        
        # Count frequencies
        for sent, tags in zip(Sents, POSs):
            prev_tag = "<START>"
            prev_prev_tag = "<START2>"
            
            for word, tag in zip(sent, tags):
                word_lower = word.lower()
                
                # Tag counts
                self.tag_counts[tag] += 1
                
                # Bigram transition counts P(tag | prev_tag)
                self.tag_tag_counts[prev_tag][tag] += 1
                
                # Trigram transition counts P(tag | prev_prev_tag, prev_tag)
                self.tag_tag_tag_counts[prev_prev_tag][prev_tag][tag] += 1
                
                # Emission counts P(word | tag)
                self.tag_word_counts[tag][word_lower] += 1
                
                # Word -> tag counts (for known words)
                self.word_tag_counts[word_lower][tag] += 1
                
                self.vocab.add(word_lower)
                
                # Suffix counts (2, 3, 4 chars)
                for suf_len in [2, 3, 4]:
                    if len(word) >= suf_len:
                        suffix = word[-suf_len:].lower()
                        self.suffix_tag_counts[suffix][tag] += 1
                
                # Prefix counts (2, 3, 4 chars)
                for pref_len in [2, 3, 4]:
                    if len(word) >= pref_len:
                        prefix = word[:pref_len].lower()
                        self.prefix_tag_counts[prefix][tag] += 1
                
                # Word shape counts
                shape = self._get_word_shape(word)
                self.shape_tag_counts[shape][tag] += 1
                
                prev_prev_tag = prev_tag
                prev_tag = tag
            
            # End of sentence
            self.tag_tag_counts[prev_tag]["<END>"] += 1
        
        self.tags = list(self.tag_counts.keys())
        self.total_words = sum(self.tag_counts.values())
        
        # Build tag index mappings
        for idx, tag in enumerate(self.tags):
            self.tag_to_idx[tag] = idx
            self.idx_to_tag[idx] = tag
        
        # Build regex patterns for unknown words
        self._build_patterns()
    
    def _build_patterns(self):
        """Build regex patterns for unknown word classification"""
        self.patterns = {
            'CD': [  # Numbers
                re.compile(r'^-?\d+(\.\d+)?$'),
                re.compile(r'^-?\d+,\d+$'),
                re.compile(r'^\d+[\d,]*\.\d+$'),
                re.compile(r'^\d+\/\d+$'),
                re.compile(r'^\d+s$'),
                re.compile(r'^\d+%$'),
                re.compile(r'^\$[\d,]+\.?\d*$'),
            ],
            'NNP': [  # Proper nouns
                re.compile(r'^[A-Z][a-z]+$'),
                re.compile(r'^[A-Z]\.$'),
            ],
            'NNPS': [  # Plural proper nouns
                re.compile(r'^[A-Z][a-z]+s$'),
            ],
            'JJ': [  # Adjectives
                re.compile(r'^.*able$'),
                re.compile(r'^.*ible$'),
                re.compile(r'^.*ful$'),
                re.compile(r'^.*ous$'),
                re.compile(r'^.*ive$'),
                re.compile(r'^.*less$'),
                re.compile(r'^.*al$'),
                re.compile(r'^.*ent$'),
                re.compile(r'^.*ant$'),
            ],
            'RB': [  # Adverbs
                re.compile(r'^.*ly$'),
            ],
            'VBG': [  # Gerunds
                re.compile(r'^.*ing$'),
            ],
            'VBN': [  # Past participles
                re.compile(r'^.*ed$'),
                re.compile(r'^.*en$'),
            ],
            'VBD': [  # Past tense
                re.compile(r'^.*ed$'),
            ],
            'NNS': [  # Plural nouns
                re.compile(r'^.*ies$'),
                re.compile(r'^.*es$'),
                re.compile(r'^[a-z]+s$'),
            ],
            'NN': [  # Nouns
                re.compile(r'^.*tion$'),
                re.compile(r'^.*sion$'),
                re.compile(r'^.*ment$'),
                re.compile(r'^.*ness$'),
                re.compile(r'^.*ity$'),
                re.compile(r'^.*er$'),
                re.compile(r'^.*or$'),
                re.compile(r'^.*ism$'),
                re.compile(r'^.*ist$'),
            ],
        }
    
    def _get_emission_prob(self, word, tag, prev_tag=None):
        """P(word | tag) with improved smoothing"""
        word_lower = word.lower()
        tag_count = self.tag_counts.get(tag, 0)
        
        if tag_count == 0:
            return 1e-10
        
        word_count = self.tag_word_counts[tag].get(word_lower, 0)
        vocab_size = len(self.vocab) + 1
        
        # Known word with this tag
        if word_count > 0:
            return (word_count + 0.01) / (tag_count + 0.01 * vocab_size)
        
        # Unknown word - use multiple features
        if word_lower not in self.vocab:
            prob = self._get_unknown_word_prob(word, tag)
            return prob
        
        # Known word but not seen with this tag - very low probability
        return 0.0001 / (tag_count + 0.0001 * vocab_size)
    
    def _get_unknown_word_prob(self, word, tag):
        """Get probability for unknown word based on features"""
        scores = []
        
        # Check regex patterns first
        if tag in self.patterns:
            for pattern in self.patterns[tag]:
                if pattern.match(word):
                    scores.append(0.35)
                    break
        
        # Word shape probabilities
        shape = self._get_word_shape(word)
        if shape in self.shape_tag_counts:
            shape_counts = self.shape_tag_counts[shape]
            shape_total = sum(shape_counts.values())
            if shape_total > 0:
                shape_prob = shape_counts.get(tag, 0) / shape_total
                if shape_prob > 0:
                    scores.append(shape_prob * 0.5)
        
        # Suffix probabilities (longer suffix = more weight)
        for suf_len in [4, 3, 2]:
            if len(word) >= suf_len:
                suffix = word[-suf_len:].lower()
                if suffix in self.suffix_tag_counts:
                    suffix_counts = self.suffix_tag_counts[suffix]
                    suffix_total = sum(suffix_counts.values())
                    if suffix_total > 0:
                        suffix_prob = suffix_counts.get(tag, 0) / suffix_total
                        if suffix_prob > 0:
                            weight = 0.4 + 0.15 * (suf_len - 2)
                            scores.append(suffix_prob * weight)
                            break
        
        # Prefix probabilities
        for pref_len in [4, 3, 2]:
            if len(word) >= pref_len:
                prefix = word[:pref_len].lower()
                if prefix in self.prefix_tag_counts:
                    prefix_counts = self.prefix_tag_counts[prefix]
                    prefix_total = sum(prefix_counts.values())
                    if prefix_total > 0:
                        prefix_prob = prefix_counts.get(tag, 0) / prefix_total
                        if prefix_prob > 0:
                            scores.append(prefix_prob * 0.25)
                            break
        
        # Capitalization features
        if word[0].isupper() and not word.isupper():
            if tag in ['NNP', 'NNPS']:
                scores.append(0.25)
        if word.isupper() and len(word) > 1:
            if tag in ['NNP', 'NNPS', 'NN']:
                scores.append(0.2)
        if any(c.isdigit() for c in word):
            if tag == 'CD':
                scores.append(0.45)
        if '-' in word and tag in ['JJ', 'NN', 'VBN']:
            scores.append(0.15)
        
        # Combine scores
        if scores:
            return max(scores)
        
        # Fallback to tag prior
        return self.tag_counts.get(tag, 1) / (self.total_words * 10)
    
    def _get_transition_prob(self, prev_tag, tag, prev_prev_tag=None):
        """P(tag | prev_tag) with interpolated trigram smoothing"""
        num_tags = len(self.tags)
        
        # Bigram probability
        prev_counts = self.tag_tag_counts.get(prev_tag, {})
        prev_total = sum(prev_counts.values()) if prev_counts else 0
        
        if prev_total == 0:
            bigram_prob = 1.0 / num_tags
        else:
            count = prev_counts.get(tag, 0)
            bigram_prob = (count + 0.01) / (prev_total + 0.01 * num_tags)
        
        # Trigram probability (if available)
        if prev_prev_tag is not None:
            trigram_counts = self.tag_tag_tag_counts.get(prev_prev_tag, {}).get(prev_tag, {})
            trigram_total = sum(trigram_counts.values()) if trigram_counts else 0
            
            if trigram_total > 5:  # Only use trigram if enough data
                trigram_count = trigram_counts.get(tag, 0)
                trigram_prob = (trigram_count + 0.01) / (trigram_total + 0.01 * num_tags)
                # Interpolate: 0.7 * trigram + 0.3 * bigram
                return 0.6 * trigram_prob + 0.4 * bigram_prob
        
        return bigram_prob
    
    def _viterbi(self, sent):
        """Second-order Viterbi algorithm"""
        n = len(sent)
        num_tags = len(self.tags)
        
        if n == 0:
            return []
        
        # For longer sentences, use beam search
        use_beam = n > 20
        beam_width = 5 if use_beam else num_tags
        
        # Initialize - track (prev_prev_tag, prev_tag) -> score
        # For first word
        viterbi_prev = {}
        for j, tag in enumerate(self.tags):
            trans_prob = self._get_transition_prob("<START>", tag, "<START2>")
            emit_prob = self._get_emission_prob(sent[0], tag)
            score = np.log(trans_prob + 1e-10) + np.log(emit_prob + 1e-10)
            viterbi_prev[("<START>", tag)] = (score, [tag])
        
        # Forward pass
        for i in range(1, n):
            word = sent[i]
            viterbi_curr = {}
            
            for j, tag in enumerate(self.tags):
                emit_prob = self._get_emission_prob(word, tag)
                log_emit = np.log(emit_prob + 1e-10)
                
                best_score = -np.inf
                best_path = None
                
                for (prev_prev_tag, prev_tag), (prev_score, path) in viterbi_prev.items():
                    trans_prob = self._get_transition_prob(prev_tag, tag, prev_prev_tag)
                    score = prev_score + np.log(trans_prob + 1e-10) + log_emit
                    
                    if score > best_score:
                        best_score = score
                        best_path = path + [tag]
                
                if best_path is not None:
                    prev_tag_for_key = best_path[-2] if len(best_path) > 1 else "<START>"
                    key = (prev_tag_for_key, tag)
                    
                    if key not in viterbi_curr or best_score > viterbi_curr[key][0]:
                        viterbi_curr[key] = (best_score, best_path)
            
            # Beam pruning
            if use_beam and len(viterbi_curr) > beam_width * num_tags:
                sorted_items = sorted(viterbi_curr.items(), key=lambda x: x[1][0], reverse=True)
                viterbi_curr = dict(sorted_items[:beam_width * num_tags])
            
            viterbi_prev = viterbi_curr
        
        # Find best final path
        best_score = -np.inf
        best_path = None
        for (prev_prev_tag, prev_tag), (score, path) in viterbi_prev.items():
            if score > best_score:
                best_score = score
                best_path = path
        
        return best_path if best_path else ['NN'] * n
    
    def predict(self, Sents):
        """Predict POS tags for sentences"""
        results = []
        for sent in Sents:
            tags = self._viterbi(sent)
            results.append(tags)
        return results

    def save(self):
        """Save the model"""
        # Convert defaultdicts to regular dicts for pickling
        model_data = {
            'tag_counts': dict(self.tag_counts),
            'tag_tag_counts': {k: dict(v) for k, v in self.tag_tag_counts.items()},
            'tag_tag_tag_counts': {k1: {k2: dict(v2) for k2, v2 in v1.items()} 
                                   for k1, v1 in self.tag_tag_tag_counts.items()},
            'tag_word_counts': {k: dict(v) for k, v in self.tag_word_counts.items()},
            'suffix_tag_counts': {k: dict(v) for k, v in self.suffix_tag_counts.items()},
            'prefix_tag_counts': {k: dict(v) for k, v in self.prefix_tag_counts.items()},
            'word_tag_counts': {k: dict(v) for k, v in self.word_tag_counts.items()},
            'shape_tag_counts': {k: dict(v) for k, v in self.shape_tag_counts.items()},
            'tags': self.tags,
            'vocab': self.vocab,
            'total_words': self.total_words,
            'tag_to_idx': self.tag_to_idx,
            'idx_to_tag': self.idx_to_tag,
        }
        with open("model.mdl", "wb") as f:
            pickle.dump(model_data, f)

    def load():
        """Load the model"""
        with open("model.mdl", "rb") as f:
            model_data = pickle.load(f)
        
        tagger = POSTagger()
        tagger.tag_counts = defaultdict(int, model_data['tag_counts'])
        
        tagger.tag_tag_counts = defaultdict(lambda: defaultdict(int))
        for k, v in model_data['tag_tag_counts'].items():
            tagger.tag_tag_counts[k] = defaultdict(int, v)
        
        tagger.tag_tag_tag_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for k1, v1 in model_data['tag_tag_tag_counts'].items():
            for k2, v2 in v1.items():
                tagger.tag_tag_tag_counts[k1][k2] = defaultdict(int, v2)
        
        tagger.tag_word_counts = defaultdict(lambda: defaultdict(int))
        for k, v in model_data['tag_word_counts'].items():
            tagger.tag_word_counts[k] = defaultdict(int, v)
            
        tagger.suffix_tag_counts = defaultdict(lambda: defaultdict(int))
        for k, v in model_data['suffix_tag_counts'].items():
            tagger.suffix_tag_counts[k] = defaultdict(int, v)
            
        tagger.prefix_tag_counts = defaultdict(lambda: defaultdict(int))
        for k, v in model_data['prefix_tag_counts'].items():
            tagger.prefix_tag_counts[k] = defaultdict(int, v)
            
        tagger.word_tag_counts = defaultdict(lambda: defaultdict(int))
        for k, v in model_data['word_tag_counts'].items():
            tagger.word_tag_counts[k] = defaultdict(int, v)
            
        tagger.shape_tag_counts = defaultdict(lambda: defaultdict(int))
        for k, v in model_data['shape_tag_counts'].items():
            tagger.shape_tag_counts[k] = defaultdict(int, v)
            
        tagger.tags = model_data['tags']
        tagger.vocab = model_data['vocab']
        tagger.total_words = model_data['total_words']
        tagger.tag_to_idx = model_data['tag_to_idx']
        tagger.idx_to_tag = model_data['idx_to_tag']
        tagger._build_patterns()
        return tagger