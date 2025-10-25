import json
from typing import List, Optional, Dict
import nltk
import re
import pymorphy3 as pym
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz

import uvicorn
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from contextlib import asynccontextmanager

import ssl

# Отключаем проверку SSL (небезопасно, использовать только для тестирования)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

class LawLink(BaseModel):
    law_id: Optional[int] = None
    article: Optional[str] = None
    point_article: Optional[str] = None
    subpoint_article: Optional[str] = None


class LinksResponse(BaseModel):
    links: List[LawLink]


class TextRequest(BaseModel):
    text: str


# Глобальные переменные для компонентов
morph_analyzer = None
stopwords_ru = None
law_aliases_invers = None


def normalize_text(text: str) -> str:
    """Нормализация текста с помощью морфологического анализа"""
    global morph_analyzer, stopwords_ru
    
    tokens = word_tokenize(text.lower(), language='russian')
    normalized_tokens = []
    
    for token in tokens:
        if re.match(r'^\d+\.\d+', token) or re.match(r'^\d+(?:\.\d+)+$', token):
            normalized_tokens.append(token)
        elif token.isdigit() or (any(c.isdigit() for c in token) and any(c.isalpha() for c in token)):
            normalized_tokens.append(token)
        elif token in ['ст', 'п', 'пп', 'ст.', 'п.', 'пп.', 'нк', 'гк', 'ук', 'тк', 'апк', 'бк', 'коап', 'рф', 'ч']:
            normalized_tokens.append(token.split('.')[0])
        elif token.isalpha() and token not in stopwords_ru:
            parsed = morph_analyzer.parse(token)[0]
            normalized_tokens.append(parsed.normal_form)
        elif token != '.':
            normalized_tokens.append(token)

    return ' '.join(normalized_tokens)


def extract_multiple_entities(raw: str):
    """Извлечение множественных сущностей из строки"""
    clean_subpoint = re.sub(r'[^\dа-я,\s\.\-]', '', raw.lower())
    clean_subpoint = re.sub(r'(?<!\d)\.(?!\d)', '', clean_subpoint)
    parts = re.split(r'[,и]', clean_subpoint)
    
    entities = []
    for part in parts:
        part = part.strip()
        if part:
            entities.append(part)
    
    return entities if entities else [raw]


def find_law_id_fuzzy(law_name):
    """Поиск ID закона по названию с помощью нечеткого поиска"""
    global law_aliases_invers
    
    max_score = 0
    for law_key in law_aliases_invers.keys():
        lev_score = fuzz.partial_ratio(law_name, law_key.lower())
        if lev_score > max_score:
            max_score = lev_score
            res_key = law_key
    if max_score < 90:
        return None
    else:
        return law_aliases_invers[res_key]


def process_match(match, context):
    """Обработка найденного совпадения и создание объектов LawLink"""
    references = []
    groups = match.groups()

    articles = []
    point_articles = []
    subpoint_articles = []

    if match.group('пункт_номера') is None:
        point_articles = [None]
        if match.group('часть_номера') is not None:
            if ',' in match.group('часть_номера') or 'и' in match.group('часть_номера'):
                point_articles = extract_multiple_entities(match.group('часть_номера'))
            else:
                point_articles = [match.group('часть_номера')]
        else:
            point_articles = [None]
    else:
        if ',' in match.group('пункт_номера') or 'и' in match.group('пункт_номера'):
            point_articles = extract_multiple_entities(match.group('пункт_номера'))
        else:
            point_articles = [match.group('пункт_номера')]

    if match.group('подпункт_номера') is None:
        subpoint_articles = [None]
    else:
        if ',' in match.group('подпункт_номера') or 'и' in match.group('подпункт_номера'):
            subpoint_articles = extract_multiple_entities(match.group('подпункт_номера'))
        else:
            subpoint_articles = [match.group('подпункт_номера')]

    if match.group('статья_номера') is None:
        articles = [None]
    else:
        if ',' in match.group('статья_номера') or 'и' in match.group('статья_номера'):
            articles = extract_multiple_entities(match.group('статья_номера'))
        else:
            articles = [match.group('статья_номера')]

    law_id = find_law_id_fuzzy(match.group('остальное'))

    for article in articles:
        for point_article in point_articles:
            for subpoint_article in subpoint_articles:
                reference = LawLink(
                    law_id=law_id,
                    article=article,
                    point_article=point_article,
                    subpoint_article=subpoint_article
                )
                references.append(reference)

    return references


def find_references_in_text(original_text: str) -> List[LawLink]:
    """Поиск правовых ссылок в тексте с помощью регулярных выражений"""
    references = []

    patterns = [
        # Паттерн для парсинга структуры ссылки
        r'(?:в\s+)?(?:(?P<подпункт_ключ>пп\.|подпункт[а-я]{0,7}|подп\.)\s*(?P<подпункт_номера>(?:\d{1,4}[а-я]?|[а-я])(?:\s*,\s*(?:\d{1,4}[а-я]?|[а-я]))*(?:\s*и\s*(?:\d{1,4}[а-я]?|[а-я]))?)\s+)?'
        r'(?:в\s+)?(?:(?P<пункт_ключ>п\.|пункт[а-я]{0,5}|пунт[а-я]{0,5})\s*(?P<пункт_номера>(?:\d{1,4}(?:[\.\-]\d{1,3})*[а-я]?|[а-я])(?:\s*,\s*(?:\d{1,4}(?:[\.\-]\d{1,3})*[а-я]?|[а-я]))*(?:\s*и\s*(?:\d{1,4}(?:[\.\-]\d{1,3})*[а-я]?|[а-я]))?)\s+)?'
        r'(?:в\s+)?(?:(?P<часть_ключ>ч\.|част[ьи])\s*(?P<часть_номера>(?:\d{1,3}(?:\.\d{1,3})?|[а-я])(?:\s*,\s*(?:\d{1,3}(?:\.\d{1,3})?|[а-я]))*(?:\s*и\s*(?:\d{1,3}(?:\.\d{1,3})?|[а-я]))?)(?:\s*,\s*)?\s+)?'
        r'(?:в\s+)?(?:(?P<статья_ключ>ст\.|стать[ейиюя]|статей?|статья)\s*(?:(?:в|на|по)\s+)?(?P<статья_номера>(?:\d{1,4}(?:[\.\-]\d{1,3})*[а-я]?)(?:\s*,\s*(?:\d{1,4}(?:[\.\-]\d{1,3})*[а-я]?))*(?:\s*и\s*(?:\d{1,4}(?:[\.\-]\d{1,3})*[а-я]?))?)\s+)?'
        r'(?P<остальное>(?:'
        # Кодексы
        r'(?:Арбитражн(?:ого|ый)\s+процессуальн(?:ого|ый)|Бюджетн(?:ого|ый)|Водн(?:ого|ый)|Воздушн(?:ого|ый)|Градостроительн(?:ого|ый)|Гражданск(?:ого|ий)|Гражданск(?:ого|ий)\s+процессуальн(?:ого|ый)|Жилищн(?:ого|ый)|Семейн(?:ого|ый)|Таможенн(?:ого|ый)|Трудов(?:ого|ой)|Уголовно-исполнительн(?:ого|ый)|Уголовно-процессуальн(?:ого|ый)|Уголовн(?:ого|ый)|Лесн(?:ого|ой)|Налогов(?:ого|ый)|Земельн(?:ого|ый))\s+кодекс(?:а|)(?:\s+Российской Федерации|\s+России|\s+РФ|)'
        # Аббревиатуры
        r'|АПК(?:\s+(?:России|РФ))?|БК(?:\s+(?:России|РФ))?|ГК(?:\s+РФ)?|ГПК(?:\s+(?:России|РФ))?|ЖК(?:\s+(?:России|РФ))?|СК(?:\s+(?:России|РФ))?|ТК(?:\s+РФ)?'
        r'|УИК(?:\s+(?:России|РФ))?|УПК(?:\s+(?:России|РФ))?|УК(?:\s+(?:России|РФ))?|ЛК(?:\s+(?:России|РФ))?|НК(?:\s+(?:России|РФ))?|ЗК(?:\s+(?:России|РФ))?'
        # Кодексы об административных правонарушениях
        r'|Кодекс(?:а|)(?:\s+Российской Федерации|\s+России|\s+РФ|)?\s+об\s+административных\s+правонарушениях|КоАП(?:\s+Российской Федерации|\s+России|\s+РФ|)?'
        # Другие кодексы
        r'|Кодекс(?:а|)\s+административного\s+судопроизводства(?:\s+Российской Федерации|\s+России|\s+РФ|)?'
        r'|Кодекс(?:а|)\s+внутреннего\s+водного\s+транспорта(?:\s+Российской Федерации|\s+России|\s+РФ|)?'
        r'|Кодекс(?:а|)\s+торгового\s+мореплавания(?:\s+Российской Федерации|\s+России|\s+РФ|)?'
        # Указы Президента
        r'|Указ(?:а|)(?:\s+Президента(?:\s+Российской Федерации|\s+России|\s+РФ|)?)?(?:\s+(?:№?\s*\d+|\s*от\s*\d{2}\.\d{2}\.\d{4}))?(?:\s*«[^»]*»)?'
        # Распоряжения Президента
        r'|Распоряжени(?:я|е)(?:\s+Президента(?:\s+Российской Федерации|\s+России|\s+РФ|)?)?(?:\s+(?:№?\s*\d+-\s*рп|от\s*\d{2}\.\d{2}\.\d{4}))?(?:\s*«[^»]*»)?'
        r'|РП(?:\s+(?:№?\s*\d+-\s*рп|от\s*\d{2}\.\d{2}\.\d{4}))?(?:\s*«[^»]*»)?'
        # Федеральные законы
        r'|Федеральн(?:ого|ый)\s+закон(?:а|)(?:\s+[^,\.;]*)?'
        r'|ФЗ(?:\s+[^,\.;]*)?'
        r'|Закона\s+«[^»]*»'
        r'|Закона\s+"[^"]*"'
        # Конституция
        r'|Конституци(?:и|я)(?:\s+РФ|\s+России|\s+Российской Федерации)?'
        # Основы законодательства
        r'|Основы\s+законодательства(?:\s+(?:Российской\s+Федерации|России|РФ))?(?:\s+№?\s*\d+(?:-I)?)?(?:\s+от\s+\d{2}\.\d{2}\.\d{4})?(?:\s*«[^»]*»)?'
        # Законы Российской Федерации
        r'|Закон(?:\s+(?:Российской\s+Федерации|России|РФ))?(?:\s+№?\s*\d+(?:-I)?)?(?:\s+от\s+\d{2}\.\d{2}\.\d{4})?(?:\s*«[^»]*»)?'
        # Федеральные стандарты бухгалтерского учета
        r'|Федеральный\s+стандарт\s+бухгалтерского\s+учета(?:\s+(?:государственных\s+финансов|для\s+организаций\s+государственного\s+сектора))?(?:\s+ФСБУ\s*\d+(?:\/\d{4})?)?(?:\s*«[^»]*»)?'
        r'|ФСБУ(?:\s+(?:государственных\s+финансов|для\s+организаций\s+государственного\s+сектора))?(?:\s*\d+(?:\/\d{4})?)?(?:\s*«[^»]*»)?'
        # Положения по бухгалтерскому учету
        r'|Положение\s+по\s+бухгалтерскому\s+учету(?:\s+ПБУ\s*\d+(?:\/\d{4})?)?(?:\s*«[^»]*»)?'
        r'|ПБУ(?:\s*\d+(?:\/\d{4})?)?(?:\s*«[^»]*»)?'
        # Положение по ведению бухгалтерского учета
        r'|Положени(?:я|е)\s+по\s+ведению\s+бухгалтерского\s+учета\s+и\s+бухгалтерской\s+отчетности\s+в\s+Российской\s+Федерации'
        r')'
        r'(?=\s|,|\.|;|$))'
    ]

    all_matches = []
    for pattern in patterns:
        matches = re.finditer(pattern, original_text, re.IGNORECASE)
        all_matches.append(matches)

    # Удаляем дубликаты и выводим
    unique_matches = list(dict.fromkeys(all_matches))

    iter_count = 0
    for matches in unique_matches:
        for match in matches:
            iter_count += 1
            print(f"Match {iter_count}: {match.groups()}")
            references.extend(process_match(match, original_text))

    return references


def extract_legal_references_advanced(text: str):
    """Основная функция для извлечения правовых ссылок из текста"""
    references = []
    sentence_references = find_references_in_text(text)
    references.extend(sentence_references)
    return references


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global morph_analyzer, stopwords_ru, law_aliases_invers
    
    print("🚀 Сервис запускается...")
    
    # Загружаем данные NLTK
    print("📥 Загружаем данные NLTK...")
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')
    except ssl.SSLError as e:
        print(f"Ошибка SSL: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    
    # Инициализируем компоненты
    print("🔧 Инициализируем компоненты...")
    morph_analyzer = pym.MorphAnalyzer()
    stopwords_ru = stopwords.words("russian")
    
    # Загружаем алиасы законов
    print("📚 Загружаем базу законов...")
    with open("law_aliases.json", "r", encoding='utf-8') as file:
        codex_aliases = json.load(file)
    
    # Создаем обратный словарь
    law_aliases_invers = {i: k for k, v in codex_aliases.items() for i in v}
    
    app.state.codex_aliases = codex_aliases
    app.state.law_aliases_invers = law_aliases_invers
    app.state.morph_analyzer = morph_analyzer
    app.state.stopwords_ru = stopwords_ru
    
    print("✅ Сервис готов к работе!")
    yield
    
    # Shutdown
    print("🛑 Сервис завершается...")
    del codex_aliases
    del law_aliases_invers
    del morph_analyzer
    del stopwords_ru


def get_codex_aliases(request: Request) -> Dict:
    return request.app.state.codex_aliases


app = FastAPI(
    title="Law Links Service",
    description="Cервис для выделения юридических ссылок из текста",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/detect")
async def get_law_links(
    data: TextRequest,
    request: Request,
    codex_aliases: Dict = Depends(get_codex_aliases),
    ) -> LinksResponse:
    """
    Принимает текст и возвращает список юридических ссылок
    """
    print(f"🔍 Обрабатываем текст длиной {len(data.text)} символов")
    
    try:
        # Извлекаем правовые ссылки из текста
        references = extract_legal_references_advanced(data.text)
        
        print(f"✅ Найдено {len(references)} правовых ссылок")
        
        return LinksResponse(links=references)
        
    except Exception as e:
        print(f"❌ Ошибка при обработке текста: {e}")
        return LinksResponse(links=[])


@app.get("/health")
async def health_check():
    """
    Проверка состояния сервиса
    """
    return {"status": "healthy"}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8978)