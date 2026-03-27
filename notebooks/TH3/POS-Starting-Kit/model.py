import pickle
from collections import Counter
import re


class POSTagger:
    """
    Structured Linear Sequence Labeler for POS Tagging
    Implements Multi-View Feature Ensemble with Viterbi Decoding
    Pure Python/NumPy implementation - no external ML libraries
    """
    
    def __init__(self):
        # Multi-view feature weights (ensemble of 3 feature perspectives)
        self.param_lexical = {}  # View 1: Word-based features
        self.param_morpho = {}   # View 2: Morphological features  
        self.param_context = {}  # View 3: Context features
        
        # Averaging accumulators
        self._acc_lex = {}
        self._acc_mor = {}
        self._acc_ctx = {}
        self._step_lex = {}
        self._step_mor = {}
        self._step_ctx = {}
        self._global_step = 0
        
        # Model metadata
        self.label_set = []
        self.frequent_mapping = {}
        self.fallback_label = 'NN'
        self.viterbi_beam = 4
        
        # Ensemble weights (learned during training)
        self.ensemble_alpha = [0.4, 0.35, 0.25]  # [lex, morpho, context]
        
    def _word_signature(self, token):
        """Generate compact word signature"""
        sig = []
        prev_type = None
        for ch in token[:10]:
            if ch.isupper():
                curr = 'U'
            elif ch.islower():
                curr = 'l'
            elif ch.isdigit():
                curr = '0'
            else:
                curr = ch
            if curr != prev_type:
                sig.append(curr)
                prev_type = curr
        return ''.join(sig)
    
    def _canonicalize(self, token):
        """Normalize token for lookup"""
        if re.match(r'^\d+$', token):
            return '<NUM>'
        if re.match(r'^-?\d+[.,]?\d*$', token):
            return '<DECIMAL>'
        return token.lower()
    
    def _extract_lexical_view(self, tokens, pos, tag_prev, tag_prev2):
        """View 1: Lexical/word-based features"""
        tok = tokens[pos]
        tok_canon = self._canonicalize(tok)
        feats = {}
        
        # Current word
        feats[f'TOK={tok_canon}'] = 1
        feats[f'SIG={self._word_signature(tok)}'] = 1
        feats[f'TLEN={min(len(tok), 12)}'] = 1
        
        # Tag history
        feats[f'P1={tag_prev}'] = 1
        feats[f'P2={tag_prev2}'] = 1
        feats[f'P12={tag_prev2}_{tag_prev}'] = 1
        
        # Tag-word combination
        feats[f'P1_TOK={tag_prev}_{tok_canon}'] = 1
        
        return feats
    
    def _extract_morpho_view(self, tokens, pos, tag_prev, tag_prev2):
        """View 2: Morphological features"""
        tok = tokens[pos]
        tok_low = tok.lower()
        feats = {}
        n = len(tok)
        
        # Case patterns
        if tok[0].isupper():
            feats['INIT_CAP'] = 1
        if tok.isupper() and n > 1:
            feats['ALL_UPPER'] = 1
        if tok.islower():
            feats['ALL_LOWER'] = 1
        if tok.istitle():
            feats['TITLECASE'] = 1
            
        # Character composition
        if any(c.isdigit() for c in tok):
            feats['CONTAINS_DIGIT'] = 1
        if '-' in tok:
            feats['CONTAINS_HYPHEN'] = 1
        if '.' in tok:
            feats['CONTAINS_PERIOD'] = 1
        if "'" in tok:
            feats['CONTAINS_APOS'] = 1
            
        # Suffixes (length 1-4)
        if n >= 2:
            feats[f'SUF1={tok_low[-1]}'] = 1
            feats[f'P1_SUF1={tag_prev}_{tok_low[-1]}'] = 1
        if n >= 3:
            feats[f'SUF2={tok_low[-2:]}'] = 1
            feats[f'P1_SUF2={tag_prev}_{tok_low[-2:]}'] = 1
        if n >= 4:
            feats[f'SUF3={tok_low[-3:]}'] = 1
            feats[f'P1_SUF3={tag_prev}_{tok_low[-3:]}'] = 1
        if n >= 5:
            feats[f'SUF4={tok_low[-4:]}'] = 1
            
        # Prefixes (length 1-3)
        if n >= 2:
            feats[f'PRE1={tok_low[0]}'] = 1
        if n >= 3:
            feats[f'PRE2={tok_low[:2]}'] = 1
        if n >= 4:
            feats[f'PRE3={tok_low[:3]}'] = 1
            
        # Common morphological endings
        endings = [
            ('ing', 'GERUND'), ('ed', 'PAST'), ('ly', 'ADVERB'),
            ('tion', 'NOMINAL'), ('sion', 'NOMINAL'), ('ness', 'NOMINAL'),
            ('ment', 'NOMINAL'), ('able', 'ADJEND'), ('ible', 'ADJEND'),
            ('ful', 'ADJEND'), ('less', 'ADJEND'), ('ous', 'ADJEND'),
            ('ive', 'ADJEND'), ('er', 'COMPAR'), ('est', 'SUPER'),
            ("'s", 'POSSESS'), ("n't", 'NEGATION')
        ]
        for ending, label in endings:
            if tok_low.endswith(ending):
                feats[f'MORPH_{label}'] = 1
                break
                
        return feats
    
    def _extract_context_view(self, tokens, pos, tag_prev, tag_prev2):
        """View 3: Context features"""
        feats = {}
        seq_len = len(tokens)
        
        # Position features
        if pos == 0:
            feats['START_POS'] = 1
        if pos == seq_len - 1:
            feats['END_POS'] = 1
            
        # Previous tokens
        if pos >= 1:
            prev_tok = self._canonicalize(tokens[pos - 1])
            feats[f'W-1={prev_tok}'] = 1
            feats[f'P1_W-1={tag_prev}_{prev_tok}'] = 1
            curr_tok = self._canonicalize(tokens[pos])
            feats[f'W-1_W0={prev_tok}_{curr_tok}'] = 1
            
        if pos >= 2:
            feats[f'W-2={self._canonicalize(tokens[pos - 2])}'] = 1
            
        # Next tokens
        if pos < seq_len - 1:
            next_tok = self._canonicalize(tokens[pos + 1])
            feats[f'W+1={next_tok}'] = 1
            curr_tok = self._canonicalize(tokens[pos])
            feats[f'W0_W+1={curr_tok}_{next_tok}'] = 1
            
        if pos < seq_len - 2:
            feats[f'W+2={self._canonicalize(tokens[pos + 2])}'] = 1
            
        return feats
    
    def _compute_score(self, params, feats):
        """Compute dot product score"""
        total = 0.0
        for f, v in feats.items():
            if f in params:
                total += params[f] * v
        return total
    
    def _ensemble_score(self, label, lex_feats, mor_feats, ctx_feats):
        """Combine scores from all views"""
        s1 = self._compute_score(self.param_lexical.get(label, {}), lex_feats)
        s2 = self._compute_score(self.param_morpho.get(label, {}), mor_feats)
        s3 = self._compute_score(self.param_context.get(label, {}), ctx_feats)
        
        return (self.ensemble_alpha[0] * s1 + 
                self.ensemble_alpha[1] * s2 + 
                self.ensemble_alpha[2] * s3)
    
    def _viterbi_decode(self, tokens):
        """Viterbi-style beam decoding"""
        if not tokens:
            return []
            
        seq_len = len(tokens)
        
        # Initialize beam: (score, tag_sequence, prev_tag, prev2_tag)
        hypotheses = [(0.0, [], '<BOS>', '<BOS2>')]
        
        for idx in range(seq_len):
            tok = tokens[idx]
            tok_lower = tok.lower()
            new_hypotheses = []
            
            # Check frequent word cache
            cached_label = self.frequent_mapping.get(tok_lower)
            
            for score, tag_seq, p1, p2 in hypotheses:
                # Extract features for all views
                lex_f = self._extract_lexical_view(tokens, idx, p1, p2)
                mor_f = self._extract_morpho_view(tokens, idx, p1, p2)
                ctx_f = self._extract_context_view(tokens, idx, p1, p2)
                
                if cached_label:
                    # Use cached label
                    label_score = self._ensemble_score(cached_label, lex_f, mor_f, ctx_f)
                    new_hypotheses.append((
                        score + label_score,
                        tag_seq + [cached_label],
                        cached_label, p1
                    ))
                else:
                    # Try all labels
                    for label in self.label_set:
                        label_score = self._ensemble_score(label, lex_f, mor_f, ctx_f)
                        new_hypotheses.append((
                            score + label_score,
                            tag_seq + [label],
                            label, p1
                        ))
            
            # Prune beam
            new_hypotheses.sort(key=lambda x: x[0], reverse=True)
            hypotheses = new_hypotheses[:self.viterbi_beam]
        
        return hypotheses[0][1] if hypotheses else [self.fallback_label] * seq_len
    
    def _greedy_decode(self, tokens):
        """Fast greedy decoding for efficiency"""
        if not tokens:
            return []
            
        result = []
        p1, p2 = '<BOS>', '<BOS2>'
        
        for idx, tok in enumerate(tokens):
            tok_lower = tok.lower()
            
            if tok_lower in self.frequent_mapping:
                chosen = self.frequent_mapping[tok_lower]
            else:
                lex_f = self._extract_lexical_view(tokens, idx, p1, p2)
                mor_f = self._extract_morpho_view(tokens, idx, p1, p2)
                ctx_f = self._extract_context_view(tokens, idx, p1, p2)
                
                best_label = self.fallback_label
                best_score = float('-inf')
                
                for label in self.label_set:
                    s = self._ensemble_score(label, lex_f, mor_f, ctx_f)
                    if s > best_score:
                        best_score = s
                        best_label = label
                
                chosen = best_label
            
            result.append(chosen)
            p2, p1 = p1, chosen
            
        return result
    
    def _label_sequence(self, tokens):
        """Choose decoding strategy based on input length"""
        if len(tokens) > 50:
            return self._greedy_decode(tokens)
        return self._viterbi_decode(tokens)
    
    def _update_params(self, params, acc, step_tracker, gold, pred, feats):
        """Averaged perceptron parameter update for one view"""
        if gold == pred:
            return
            
        self._global_step += 1
        
        # Initialize label dicts if needed
        if gold not in params:
            params[gold] = {}
            acc[gold] = {}
            step_tracker[gold] = {}
        if pred not in params:
            params[pred] = {}
            acc[pred] = {}
            step_tracker[pred] = {}
            
        for f in feats:
            # Update gold label (increase weight)
            if f not in params[gold]:
                params[gold][f] = 0.0
                acc[gold][f] = 0.0
                step_tracker[gold][f] = 0
            acc[gold][f] += (self._global_step - step_tracker[gold][f]) * params[gold][f]
            step_tracker[gold][f] = self._global_step
            params[gold][f] += 1.0
            
            # Update predicted label (decrease weight)
            if f not in params[pred]:
                params[pred][f] = 0.0
                acc[pred][f] = 0.0
                step_tracker[pred][f] = 0
            acc[pred][f] += (self._global_step - step_tracker[pred][f]) * params[pred][f]
            step_tracker[pred][f] = self._global_step
            params[pred][f] -= 1.0
    
    def _finalize_averaging(self, params, acc, step_tracker):
        """Apply averaged weights"""
        for label in list(params.keys()):
            for f in list(params[label].keys()):
                total = acc[label][f]
                total += (self._global_step - step_tracker[label][f]) * params[label][f]
                avg_weight = total / self._global_step if self._global_step > 0 else 0
                
                if abs(avg_weight) < 1e-4:
                    del params[label][f]
                else:
                    params[label][f] = avg_weight
    
    def fit(self, Sents, POSs, n_iter=10):
        """Train the multi-view ensemble tagger"""
        print(f"Training Multi-View Ensemble Tagger ({n_iter} epochs)...")
        
        # Collect statistics
        word_label_freq = {}
        label_counter = Counter()
        
        for sent, labels in zip(Sents, POSs):
            for tok, lbl in zip(sent, labels):
                tok_low = tok.lower()
                if tok_low not in word_label_freq:
                    word_label_freq[tok_low] = Counter()
                word_label_freq[tok_low][lbl] += 1
                label_counter[lbl] += 1
        
        self.label_set = list(label_counter.keys())
        self.fallback_label = label_counter.most_common(1)[0][0]
        
        # Build frequent word cache (high confidence words)
        for tok_low, lbl_counts in word_label_freq.items():
            total = sum(lbl_counts.values())
            if total >= 12:
                dominant_lbl, dominant_cnt = lbl_counts.most_common(1)[0]
                if dominant_cnt / total > 0.96:
                    self.frequent_mapping[tok_low] = dominant_lbl
        
        print(f"  Labels: {len(self.label_set)}, Cached words: {len(self.frequent_mapping)}")
        
        # Initialize parameter dicts
        for label in self.label_set:
            self.param_lexical[label] = {}
            self.param_morpho[label] = {}
            self.param_context[label] = {}
            self._acc_lex[label] = {}
            self._acc_mor[label] = {}
            self._acc_ctx[label] = {}
            self._step_lex[label] = {}
            self._step_mor[label] = {}
            self._step_ctx[label] = {}
        
        import random
        indices = list(range(len(Sents)))
        
        # Training loop
        for epoch in range(n_iter):
            random.shuffle(indices)
            hits, total = 0, 0
            
            for i in indices:
                sentence = Sents[i]
                gold_labels = POSs[i]
                p1, p2 = '<BOS>', '<BOS2>'
                
                for pos, (tok, gold_lbl) in enumerate(zip(sentence, gold_labels)):
                    # Extract all view features
                    lex_f = self._extract_lexical_view(sentence, pos, p1, p2)
                    mor_f = self._extract_morpho_view(sentence, pos, p1, p2)
                    ctx_f = self._extract_context_view(sentence, pos, p1, p2)
                    
                    # Predict
                    pred_lbl = self.fallback_label
                    max_score = float('-inf')
                    for label in self.label_set:
                        s = self._ensemble_score(label, lex_f, mor_f, ctx_f)
                        if s > max_score:
                            max_score = s
                            pred_lbl = label
                    
                    # Update all view parameters
                    self._update_params(
                        self.param_lexical, self._acc_lex, self._step_lex,
                        gold_lbl, pred_lbl, lex_f
                    )
                    self._update_params(
                        self.param_morpho, self._acc_mor, self._step_mor,
                        gold_lbl, pred_lbl, mor_f
                    )
                    self._update_params(
                        self.param_context, self._acc_ctx, self._step_ctx,
                        gold_lbl, pred_lbl, ctx_f
                    )
                    
                    if pred_lbl == gold_lbl:
                        hits += 1
                    total += 1
                    
                    p2, p1 = p1, gold_lbl
            
            acc = hits / total * 100
            print(f"  Epoch {epoch + 1}/{n_iter}: {acc:.2f}%")
        
        # Finalize averaging
        self._finalize_averaging(self.param_lexical, self._acc_lex, self._step_lex)
        self._finalize_averaging(self.param_morpho, self._acc_mor, self._step_mor)
        self._finalize_averaging(self.param_context, self._acc_ctx, self._step_ctx)
        
        # Cleanup training accumulators
        self._acc_lex = self._acc_mor = self._acc_ctx = None
        self._step_lex = self._step_mor = self._step_ctx = None
        
        total_params = sum(
            len(self.param_lexical.get(l, {})) +
            len(self.param_morpho.get(l, {})) +
            len(self.param_context.get(l, {}))
            for l in self.label_set
        )
        print(f"Training complete! Total parameters: {total_params}")
    
    def predict(self, Sents):
        """Predict POS tags for sentences"""
        return [self._label_sequence(sent) for sent in Sents]
    
    def save(self):
        """Persist model to disk"""
        data = {
            'lex': self.param_lexical,
            'mor': self.param_morpho,
            'ctx': self.param_context,
            'labels': self.label_set,
            'freq_map': self.frequent_mapping,
            'default': self.fallback_label,
            'beam': self.viterbi_beam,
            'alpha': self.ensemble_alpha,
        }
        with open("model.mdl", "wb") as fp:
            pickle.dump(data, fp)
    
    @staticmethod
    def load():
        """Restore model from disk"""
        with open("model.mdl", "rb") as fp:
            data = pickle.load(fp)
        
        tagger = POSTagger()
        tagger.param_lexical = data['lex']
        tagger.param_morpho = data['mor']
        tagger.param_context = data['ctx']
        tagger.label_set = data['labels']
        tagger.frequent_mapping = data['freq_map']
        tagger.fallback_label = data['default']
        tagger.viterbi_beam = data.get('beam', 4)
        tagger.ensemble_alpha = data.get('alpha', [0.4, 0.35, 0.25])
        
        return tagger