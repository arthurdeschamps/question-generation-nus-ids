from abc import ABC
from typing import List, Dict
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


class SquadExample(QAExample, JsonParsable):

    def __init__(self, context: str, *args, **kwargs):
        super(SquadExample, self).__init__(*args, **kwargs)
        self.context = context

    @staticmethod
    def from_json(json, break_up_paragraphs=True) -> List['SquadExample']:
        squad_examples = []
        for paragraph in json['paragraphs']:
            sentences_bounds = SquadExample.get_sentences_bounds(paragraph['context'])
            for qa in paragraph['qas']:
                start_indices = set()
                texts = set()
                answers = qa['answers']
                for answer_json in answers:
                    answer = Answer.from_json(answer_json)
                    if answer.answer_start not in start_indices or answer.text not in texts:
                        start_indices.add(answer.answer_start)
                        texts.add(answer.text)
                        question = Question.from_json(qa)
                        if break_up_paragraphs:
                            context = SquadExample.get_answer_context(answer, sentences_bounds)
                        else:
                            context = paragraph['context']
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
