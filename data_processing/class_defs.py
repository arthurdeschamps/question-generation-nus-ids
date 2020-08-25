from abc import ABC
from typing import List, Dict, Optional
import nltk


nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


class JsonParsable(ABC):
    @staticmethod
    def from_json(json):
        raise NotImplementedError()


class Question(JsonParsable):

    def __init__(self, question: str, question_id: str):
        super(Question, self).__init__()
        self.question = question
        self.question_id = question_id

    @staticmethod
    def from_json(json) -> 'Question':
        q = Question(json['question'], json['id'])
        try:
            question_mark_ind = str.index(q.question, "?")
        except ValueError:
            question_mark_ind = len(q.question) - 1
        if q.question[question_mark_ind - 1] != ' ':
            q.question = q.question[:question_mark_ind] + ' ' + '?'
        return q


class Answer(JsonParsable):

    def __init__(self, text: str, answer_start: int):
        super(Answer, self).__init__()
        self.text = text
        self.answer_start = answer_start

    @staticmethod
    def from_json(json) -> 'Answer':
        return Answer(json['text'], json['answer_start'])


class QAExample:

    def __init__(self, question: Question, answer: Answer):
        super(QAExample, self).__init__()
        self.question = question
        self.answer = answer


class QAsExample:

    def __init__(self, question: Question, answers: List[Answer]):
        super(QAsExample, self).__init__()
        self.question = question
        self.answers = answers


class SquadMultiQAExample(JsonParsable):

    def __init__(self, context: str, qas: List[QAsExample], *args, **kwargs):
        super(SquadMultiQAExample, self).__init__(*args, **kwargs)
        self.context = context
        self.qas = qas

    @staticmethod
    def from_json(json):
        squad_multi_examples = []
        for paragraph in json['paragraphs']:
            context = paragraph['context']
            qas = []
            for qa in paragraph['qas']:
                answers = [Answer.from_json(answer) for answer in qa['answers']]
                question = Question.from_json(qa)
                qas.append(QAsExample(question, answers))
            squad_multi_examples.append(SquadMultiQAExample(context, qas))
        return squad_multi_examples


class RepeatQFeature:

    def __init__(self, pos_tags, entity_tags, ner, letter_cases):
        super(RepeatQFeature, self).__init__()
        self.pos_tags = pos_tags
        self.entity_tags = entity_tags
        self.ner = ner
        self.letter_cases = letter_cases


class RepeatQExample(JsonParsable):

    def __init__(self,
                 base_question: str,
                 base_question_features: str,
                 facts: List[str],
                 facts_features: List[str],
                 rephrased_question: Optional[str],
                 passage_id: int,
                 *args, **kwargs):
        super(RepeatQExample, self).__init__(*args, **kwargs)
        self.base_question = base_question
        self.base_question_features = base_question_features
        self.facts = facts
        self.facts_features = facts_features
        self.rephrased_question = rephrased_question
        self.passage_id = passage_id

    @staticmethod
    def from_json(json) -> List['RepeatQExample']:
        return [RepeatQExample(
            base_question=example["base_question"],
            base_question_features=RepeatQFeature(
                pos_tags=example["base_question_pos_tags"], entity_tags=example["base_question_entity_tags"],
                ner=example["base_question_ner"], letter_cases=example["base_question_letter_cases"]
            ),
            facts=example["facts"],
            facts_features=[RepeatQFeature(
                pos_tags=pos, entity_tags=entity, ner=ner, letter_cases=cases
            ) for pos, entity, ner, cases in zip(example["facts_pos_tags"], example["facts_entity_tags"],
                                                 example["facts_ner"], example["facts_letter_cases"])],
            rephrased_question=example["target"],
            passage_id=example["passage_id"]
        ) for example in json]


class SquadExample(QAExample, JsonParsable):

    def __init__(self, context: str, *args, **kwargs):
        super(SquadExample, self).__init__(*args, **kwargs)
        self.context = context

    @staticmethod
    def from_json(json) -> List['SquadExample']:
        squad_examples = []
        for paragraph in json['paragraphs']:
            sentences_bounds = SquadExample.get_sentences_bounds(paragraph['context'])
            for qa in paragraph['qas']:
                start_indices = set()
                texts = set()
                answers = qa['answers']
                question = Question.from_json(qa)
                for answer_json in answers:
                    answer = Answer.from_json(answer_json)
                    if answer.answer_start not in start_indices or answer.text not in texts:
                        start_indices.add(answer.answer_start)
                        texts.add(answer.text)
                        context = SquadExample.get_answer_context(answer, sentences_bounds)
                        squad_examples.append(SquadExample(context, question, answer))
        return squad_examples

    @staticmethod
    def get_answer_context(answer: Answer, sentences_bounds: Dict[int, str]):
        sentence_start = 0
        for bound in sentences_bounds.keys():
            if answer.answer_start < bound:
                return sentences_bounds[bound]
            sentence_start = bound
        return sentences_bounds[sentence_start]

    @staticmethod
    def get_sentences_bounds(context: str) -> Dict[int, str]:
        sentences = tokenizer.tokenize(context)
        ind = 0
        bounds = {}
        for sentence in sentences:
            ind += len(sentence)
            bounds[ind] = sentence
        return bounds
