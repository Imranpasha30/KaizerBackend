import logging
from pipeline_v2.models import Stage1Output, Stage1_5Output, WordCut

logger = logging.getLogger("pipeline_v2.stage_1_5")

class Stage1_5WordCut:
    def process(self, stage1: Stage1Output) -> Stage1_5Output:
        logger.info(f"Stage 1.5 parsing {len(stage1.transcript.words)} words for garbage extraction.")
        
        # Convert Word models to WordCut
        words = []
        for w in stage1.transcript.words:
            # We must handle Pydantic properly. Since WordCut inherits from Word,
            # we can pass the fields.
            words.append(WordCut(**w.model_dump()))
            
        if not words:
            return Stage1_5Output(word_cuts=words)
            
        # --- Heuristic 1: Pre-roll Warm-up Chatter ---
        # Look at words in the first 15 seconds. Group into bursts separated by > 1.5s
        blocks = []
        current_block = []
        for i, wc in enumerate(words):
            if wc.s > 15.0:
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                break # only looking at the start
                
            if not current_block:
                current_block.append(i)
            else:
                prev_wc = words[current_block[-1]]
                if wc.s - prev_wc.e > 1.5:
                    blocks.append(current_block)
                    current_block = [i]
                else:
                    current_block.append(i)
                    
        if current_block:
            blocks.append(current_block)
            
        # If the first block is short (< 5 words), mark it as warm_up_chatter
        if blocks:
            first_block = blocks[0]
            if len(first_block) < 5:
                logger.info(f"Found pre-roll warm up chatter: {[words[i].w for i in first_block]}")
                for idx in first_block:
                    words[idx].exclude = True
                    words[idx].exclude_reason = "warm_up_chatter"

        # --- Heuristic 2: Known Filler Words ---
        # A mix of English, Hindi, and Telugu filler words.
        fillers = {
            "uh", "um", "ah", "okay", "ok", "so", "like", 
            "అ", "మరి", "ఆ", "ready", "good", "start", "యా", "ఆహ్"
        }
        
        for wc in words:
            # skip if already excluded
            if wc.exclude:
                continue
                
            # clean punctuation
            clean_w = wc.w.lower().strip(".,?!\"'")
            if clean_w in fillers:
                wc.exclude = True
                wc.exclude_reason = "filler"

        excluded_count = sum(1 for wc in words if wc.exclude)
        logger.info(f"Stage 1.5 completed. Marked {excluded_count} words for exclusion.")
        
        return Stage1_5Output(word_cuts=words)
