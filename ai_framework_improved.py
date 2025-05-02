import random
import json
import logging
import jieba
from difflib import SequenceMatcher
import unittest
import torch
import torch.quantization
import torch.nn.utils.prune as prune
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from enum import Enum
import functools
import asyncio
import time


class Config:
    DATA_FILE = "emotion_state.json"
    LOG_FILE = "emotion_logs.log"
    SIMILARITY_THRESHOLD = 0.6
    MAX_HISTORY_LENGTH = 5
    STOP_WORDS = {"的", "了", "呢", "啊", "呀", "哦", "吧"}


class LoggingInitializer:
    def __init__(self):
        logging.basicConfig(
            filename=Config.LOG_FILE,
            level=logging.INFO,
            format="%(asctime)s - %(emotion)s - %(action)s - 当前值: %(value)s - 变化: %(delta)s - 特殊状态: %(state)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            style="%"
        )


class ModelLoader:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-chinese')
        self.model = AutoModel.from_pretrained('distilbert-base-chinese')
        self._quantize_model()
        self._prune_model()
        self.student_model = self._create_student_model()
        self._knowledge_distillation()

    def _quantize_model(self):
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )

    def _prune_model(self):
        for name, module in self.quantized_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0.2)
                prune.remove(module, 'weight')

    def _create_student_model(self):
        class StudentModel(nn.Module):
            def __init__(self):
                super(StudentModel, self).__init__()
                self.embedding = nn.Embedding(self.tokenizer.vocab_size, 128)
                self.fc = nn.Linear(128, 768)

            def forward(self, input_ids):
                embedded = self.embedding(input_ids)
                output = self.fc(embedded.mean(dim=1))
                return output

        return StudentModel()

    def _knowledge_distillation(self):
        texts = [
            "这是一个测试句子",
            "另一个测试句子",
            "今天天气很不错",
            "这个电影非常精彩",
            "学习是一件有趣的事情",
            "工作让我感到充实",
            "朋友是生活中的财富",
            "美食能治愈一切烦恼",
            "运动可以保持健康",
            "阅读能开阔视野"
        ]
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.student_model.parameters(), lr=1e-3)
        for epoch in range(10):
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt')
                with torch.no_grad():
                    teacher_output = self.quantized_model(**inputs).last_hidden_state.mean(dim=1)
                student_output = self.student_model(inputs['input_ids'])
                loss = criterion(student_output, teacher_output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


class SimilarityCalculator:
    def __init__(self, model_loader):
        self.model_loader = model_loader

    @functools.lru_cache(maxsize=128)
    def sequence_matcher_similarity(self, text1: str, text2: str) -> float:
        return SequenceMatcher(None, text1, text2).ratio()

    @functools.lru_cache(maxsize=128)
    async def bert_similarity(self, text1: str, text2: str) -> float:
        max_length = 128
        inputs1 = self.model_loader.tokenizer(text1, return_tensors='pt', max_length=max_length, truncation=True)
        inputs2 = self.model_loader.tokenizer(text2, return_tensors='pt', max_length=max_length, truncation=True)
        loop = asyncio.get_running_loop()
        outputs1 = await loop.run_in_executor(None, lambda: self.model_loader.student_model(inputs1['input_ids']))
        outputs2 = await loop.run_in_executor(None, lambda: self.model_loader.student_model(inputs2['input_ids']))
        embeddings1 = outputs1.detach().numpy()
        embeddings2 = outputs2.detach().numpy()
        return cosine_similarity(embeddings1, embeddings2)[0][0]

    async def combined_similarity(self, text1: str, text2: str, threshold=Config.SIMILARITY_THRESHOLD) -> float:
        if len(text1) <= 3 or len(text2) <= 3:
            return self.sequence_matcher_similarity(text1, text2)
        seq_sim = self.sequence_matcher_similarity(text1, text2)
        if seq_sim < threshold:
            return seq_sim
        else:
            return await self.bert_similarity(text1, text2)


class SpecialState(Enum):
    FROWNING = ("-12.5 <= calm < 0", "75 <= favor < 125", 0.5)
    COQUETISH = ("-12.5 <= calm < 0", "favor >= 125", 0.75)
    DAZED = ("25 <= happiness <= 50", "50 <= favor <= 75", "-10 <= calm <= 10")
    ENVIOUS = ("50 <= happiness <= 75", "100 <= favor <= 125", "10 <= calm <= 30")
    BLUSHING = ("25 <= happiness <= 40", "75 <= favor <= 100", "-20 <= calm <= -5")
    SAD = ("-50 <= happiness <= -25", "25 <= favor <= 50", "-30 <= calm <= -10")
    SMILING = ("50 <= happiness <= 75", "75 <= favor <= 100", "10 <= calm <= 25")
    ANGRY = ("-25 <= happiness <= 0", "-25 <= favor <= 0", "-15 <= calm <= -5")
    WEARY = ("-50 <= happiness <= -30", "-50 <= favor <= -25", "-50 <= calm <= -30")
    STARTLED = ("50 <= happiness <= 75", "50 <= favor <= 75", "30 <= calm <= 50")
    LOST = ("-25 <= happiness <= 0", "50 <= favor <= 75", "0 <= calm <= 10")
    CHUCKLING = ("80 <= happiness <= 100", "125 <= favor <= 150", "20 <= calm <= 40")


class EmotionDecayManager:
    def __init__(self, emotion_manager):
        self.emotion_manager = emotion_manager

    def decay_emotions(self):
        elapsed = (time.time() - self.emotion_manager.last_interaction) / 3600
        if elapsed < 2.4:
            return

        h_level = self.emotion_manager.get_happiness_level()
        f_level = self.emotion_manager.get_favor_level()
        c_level = self.emotion_manager.get_calm_level()

        if h_level in ["高兴", "开心", "伤心", "悲伤"]:
            h_decay = 5 / 24 if h_level == "高兴" else (10 / 24 if h_level == "开心" else (4 / 24 if h_level == "伤心" else 3.5 / 24))
            self.emotion_manager._update("favor", -h_decay * elapsed, "自然降低")

        if f_level in ["心动区间", "喜欢", "关心", "害羞", "反感", "厌恶"]:
            f_decay = 5 / 24 if f_level == "心动区间" else (7.5 / 24 if f_level == "喜欢" else (10 / 24 if f_level == "关心" else (12.5 / 24 if f_level == "害羞" else (-7.5 / 24 if f_level == "反感" else -5 / 24))))
            self.emotion_manager._update("favor", f_decay * elapsed, "自然变化")

        if c_level in ["生气", "愤怒"] and elapsed >= 36:
            c_increase = 10 / 24 if c_level == "生气" else 7.5 / 24
            self.emotion_manager._update("calm", c_increase * elapsed, "自然恢复")

        if c_level == "平静" and self.emotion_manager.calm < 50:
            self.emotion_manager._update("calm", min(50 - self.emotion_manager.calm, elapsed * 0.2), "自然放松")


class EmotionManager:
    def __init__(self, similarity_calculator):
        self.happiness = 10
        self.favor = 0
        self.calm = 15
        self.last_interaction = time.time()
        self.happiness_history = []
        self.favor_history = []
        self.similarity_calculator = similarity_calculator
        self.decay_manager = EmotionDecayManager(self)
        self.load_state()
        self.happiness_history.append(())
        self.favor_history.append(())

    async def is_same_evaluation(self, current, history):
        current_text = ''.join(current)
        history_text = ''.join(history) if history else ''
        return await self.similarity_calculator.combined_similarity(current_text, history_text) >= Config.SIMILARITY_THRESHOLD

    def extract_keywords(self, text):
        return [word for word in jieba.cut(text) if word.strip() and word not in Config.STOP_WORDS]

    def save_state(self):
        state = {
            "happiness": self.happiness,
            "favor": self.favor,
            "calm": self.calm,
            "last_interaction": self.last_interaction,
            "happiness_history": self.happiness_history[-Config.MAX_HISTORY_LENGTH:],
            "favor_history": self.favor_history[-Config.MAX_HISTORY_LENGTH:]
        }
        with open(Config.DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self):
        try:
            with open(Config.DATA_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
                self.happiness = state["happiness"]
                self.favor = state["favor"]
                self.calm = state["calm"]
                self.last_interaction = state["last_interaction"]
                self.happiness_history = state["happiness_history"] or [()]
                self.favor_history = state["favor_history"] or [()]
        except FileNotFoundError:
            pass

    def _log(self, emotion, action, value, delta):
        state = self.check_special_state()
        extra = {"emotion": emotion, "action": action, "value": value, "delta": delta, "state": state}
        logging.info("", extra=extra)

    def get_happiness_level(self):
        h = self.happiness
        return ("高兴" if h >= 75 else "开心" if h >= 25 else "愉快" if h >= 0 else "伤心" if h >= -25 else "悲伤")

    def get_favor_level(self):
        f = self.favor
        return ("心动区间" if f >= 125 else "喜欢" if f >= 100 else "关心" if f >= 50 else "害羞" if f >= 25 else "触动" if f > 0 else "反感" if f >= -25 else "厌恶")

    def get_calm_level(self):
        return "平静" if self.calm >= 0 else "生气" if self.calm >= -25 else "愤怒"

    def process_praise(self, emotion, text):
        if len(text) <= 3:
            delta = 5
            self._update(emotion, delta, "短句赞美")
            return
        keywords = self.extract_keywords(text)
        history = self.happiness_history if emotion == "happiness" else self.favor_history
        is_same = asyncio.run(self.is_same_evaluation(keywords, history[-1][1] if history[-1] else []))
        base = 15 if emotion == "happiness" else 10
        rate = 0.75 if is_same else (0.8 if emotion == "happiness" else 0.85)
        prev_rate = history[-1][0] if history[-1] else 1.0
        delta = max(5, min(25, round(base * prev_rate * rate, 2)))
        self._update(emotion, delta, "增长")
        history.append((prev_rate * rate, keywords))
        history = history[-Config.MAX_HISTORY_LENGTH:]

    def process_criticism(self, emotion, text):
        base = 20
        if emotion == "happiness":
            rate = 1.25
        elif emotion == "favor":
            rate = 1.0 + 0.1 * sum(1 for e in self.favor_history if e[0] < 0)
        else:
            rate = 1.5
        delta = max(5, min(40, round(base * rate, 2)))
        self._update(emotion, -delta, "降低")

    def _update(self, emotion, delta, action):
        attr = f"{emotion}"
        max_val = {"happiness": 100, "favor": 150, "calm": 50}[emotion]
        current_val = self.__dict__[attr]
        max_delta = abs(current_val * 0.2) if current_val != 0 else 20
        delta = max(-max_delta, min(max_delta, delta))
        self.__dict__[attr] = max(-50, min(max_val, self.__dict__[attr] + delta))
        self.last_interaction = time.time()
        self._log(emotion, action, self.__dict__[attr], delta)
        self.save_state()

    def decay_emotions(self):
        self.decay_manager.decay_emotions()

    def check_special_state(self):
        c, f, h = self.calm, self.favor, self.happiness
        for state in SpecialState:
            conditions = state.value
            if len(conditions) == 3 and isinstance(conditions[2], float):
                if eval(conditions[0]) and eval(conditions[1]) and random.random() < conditions[2]:
                    return state.name
            elif eval(conditions[0]) and eval(conditions[1]) and eval(conditions[2]):
                return state.name
        return None


class TestEmotionManager(unittest.TestCase):
    def setUp(self):
        logging_initializer = LoggingInitializer()
        model_loader = ModelLoader()
        similarity_calculator = SimilarityCalculator(model_loader)
        self.em = EmotionManager(similarity_calculator)

    def test_get_favor_level(self):
        self.assertEqual(self.em.get_favor_level(), "反感" if self.em.favor == 0 else "触动")

    def test_process_praise(self):
        initial_happiness = self.em.happiness
        self.em.process_praise("happiness", "你好棒")
        self.assertTrue(self.em.happiness > initial_happiness)

    def test_process_criticism(self):
        initial_calm = self.em.calm
        self.em.process_criticism("calm", "你错了")
        self.assertTrue(self.em.calm < initial_calm)

    def test_check_special_state_boundary(self):
        self.em.happiness = 25
        self.em.favor = 50
        self.em.calm = -10
        state = self.em.check_special_state()
        if state:
            self.assertEqual(state, SpecialState.DAZED.name)

    def test_extreme_value_control(self):
        self.em._update("happiness", 100, "测试")
        self.assertEqual(self.em.happiness, 100)
        self.em._update("calm", -60, "测试")
        self.assertEqual(self.em.calm, -50)


if __name__ == '__main__':
    unittest.main()

