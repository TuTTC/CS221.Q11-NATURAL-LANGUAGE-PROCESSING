import pickle
import random
import math
from collections import defaultdict, Counter

class KneserNeyLM:
    def __init__(self, discount=0.75):
        self.order = 2 # [2-GRAM] Bigram
        self.discount = discount
        
        # Cấu trúc đếm Kneser-Ney
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = defaultdict(int)
        self.continuation_counts = defaultdict(int)
        
        self.total_unique_bigrams = 0
        self.vocab = set()
        self.starts = []
        
        self.default_filename = "Model.mdl" 

    def fit(self, data):
        seen_continuations = set() 
        
        for line in data:
            tokens = line.strip().split()
            if len(tokens) < 2: continue
            
            self.starts.append((tokens[0],))
            self.vocab.update(tokens)
            
            for i in range(len(tokens) - 1):
                context = (tokens[i],)   # Context: (w_i)
                word = tokens[i+1]       # Target: w_i+1
                
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1
                
                bg_key = (context, word)
                if bg_key not in seen_continuations:
                    self.continuation_counts[word] += 1
                    seen_continuations.add(bg_key)
                    self.total_unique_bigrams += 1

    def _get_prob_dist(self, context):
        candidates = set(self.ngram_counts[context].keys())
        
        if self.context_counts[context] == 0:
            return self._get_continuation_dist()

        unique_followers = len(candidates)
        lambda_val = (self.discount / self.context_counts[context]) * unique_followers
        
        probs = {}
        total_prob = 0
        
        targets = candidates.union(self._get_top_continuations(10))

        for w in targets:
            count = self.ngram_counts[context][w]
            first_term = max(count - self.discount, 0) / self.context_counts[context]
            
            p_cont = self.continuation_counts[w] / self.total_unique_bigrams if self.total_unique_bigrams > 0 else 0
            
            p_kn = first_term + (lambda_val * p_cont)
            probs[w] = p_kn
            total_prob += p_kn
            
        return probs, total_prob

    def _get_continuation_dist(self):
        probs = {}
        total = 0
        if self.total_unique_bigrams == 0: return {}, 0
        
        for w in self._get_top_continuations(50):
            p = self.continuation_counts[w] / self.total_unique_bigrams
            probs[w] = p
            total += p
        return probs, total

    def _get_top_continuations(self, n=20):
        return [w for w, c in sorted(self.continuation_counts.items(), key=lambda x: x[1], reverse=True)[:n]]

    def generate(self):
        if not self.starts: return ""
        
        current_state = random.choice(self.starts) # Tuple 1 phần tử
        ret = [current_state[0]]
        
        MAXLEN = 50
        for _ in range(MAXLEN):
            probs, total_prob = self._get_prob_dist(current_state)
            
            if total_prob == 0: break
            
            words = list(probs.keys())
            weights = list(probs.values())
            
            next_word = random.choices(words, weights=weights, k=1)[0]
            
            if next_word == "<END>": break
            
            ret.append(next_word)
            
            current_state = (next_word,)
            
        return " ".join(ret)
    
    def generate_with_temp(self, temperature=1.0):
        """
        Generates text with temperature scaling.
        Temp < 1.0: Conservative (picks high prob words).
        Temp > 1.0: Creative/Random (flattens distribution).
        """
        if not self.starts: return ""
        
        current_state = random.choice(self.starts) 
        ret = [current_state[0]]
        
        MAXLEN = 50
        for _ in range(MAXLEN):
            probs, total_prob = self._get_prob_dist(current_state)
            
            if total_prob == 0: break
            
            words = list(probs.keys())
            
            raw_weights = list(probs.values())
            
            if temperature != 1.0:
                adjusted_weights = [math.pow(max(w, 1e-10), 1.0/temperature) for w in raw_weights]
            else:
                adjusted_weights = raw_weights
            
            next_word = random.choices(words, weights=adjusted_weights, k=1)[0]
            
            if next_word == "<END>": break
            
            ret.append(next_word)
            current_state = (next_word,)
            
        return " ".join(ret)

    def evaluate_accuracy(self, classifier_func, target_label, n_samples=100, temperature=1.0):
        """
        Generates n_samples, runs them through the classifier_func, 
        and calculates the percentage that match target_label.
        
        classifier_func: Function that takes text -> returns label/class
        target_label: The desired output (e.g., 'Positive', 1, True)
        """
        matches = 0
        valid_samples = 0
        
        for _ in range(n_samples):
            text = self.generate_with_temp(temperature=temperature)
            
            if not text.strip(): continue
            
            valid_samples += 1
            
            try:
                prediction = classifier_func(text)
                if prediction == target_label:
                    matches += 1
            except Exception as e:
                print(f"Classifier failed on text: {text[:20]}... Error: {e}")
                
        if valid_samples == 0: return 0.0
        
        return matches / valid_samples
    

    def save(self, filename=None):
        if filename is None:
            filename = self.default_filename
        with open(filename, "wb") as f:
            pickle.dump(self, f)


# --- CLASS CON ---

class FirstLM(KneserNeyLM):
    def __init__(self):
        super().__init__(discount=0.4)
        self.default_filename = "FirstLM.mdl"

class SecondLM(KneserNeyLM):
    def __init__(self):
        super().__init__(discount=0.6)
        self.default_filename = "SecondLM.mdl"