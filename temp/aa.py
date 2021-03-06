import json

import requests


def call(product_names):
    result = dict()
    with requests.Session() as session:
        for product_name in product_names:
            search_result_names: list = result.get(product_name, [])
            headers = {
                "X-Naver-Client-Id": "XmHwgwdBJWZbpdCkn3aL",
                "X-Naver-Client-Secret": "Vr_6qJ_ukv",
            }
            url = f"https://openapi.naver.com/v1/search/shop.json?query={product_name}&display=40&sort=sim"
            response = session.get(url, headers=headers)
            data = response.json()
            for item in data.get("items", []):
                search_result_names.append(item.get("title"))
            result[product_name] = search_result_names
    with open("result.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    product_names = [
        "세차미트 워시미트 세차 타월 타올 걸레 세차장갑",
        "욕실의자 접이식 휴대용 의자 캠핑 낚시 간이 의자",
        "남성 가죽 장갑 스마트폰 터치 두꺼운 겨울 방한장갑",
        "차량용 뒷자석 핸드폰 거치대 휴대폰 헤드레스트 홀더",
        "가죽 양털장갑 오토바이장갑 스마트폰장갑 등산장갑",
        "F2288246 스웨이드 블로퍼",
        "캠핑 랜턴 감성 LED 조명 손전등 비상조명등",
        "간편먼지털이개 (대)",
        "다기능 전기약탕기 2.0L 차망포함",
        "간편 유리닦이2단 S1SET",
        "언제나위생행주 3p",
        "아트 글라스 유리물병 1000ml flower2",
        "간편 벨크로청소기 1p",
        "초극세사 (대) 밀대 1p",
        "도모스텐손잡이텀블러(블랙) 590ML",
        "코너보호대 무당벌레 2p 핑크",
        "간편 벨크로청소기 리필2P",
        "에어프라이어 종이 호일 16cm 30매",
        "간편먼지털이개 총채",
        "쉐프온(헤) 부직포수세미 10매입",
        "다기능 전기약탕기 2.0L",
        "간편샌들 브라운 270mm",
        "옻칠목기 핑크",
        "내츄럴타올4p",
        "순면행주 3p",
        "아띠홈 오토디스펜서 거품형 250ml 화이트",
        "미니쿠션 2m+코너 4p 베이지",
        "옻칠목기 민트",
        "빈티지 다용도 우산꽂이 영문",
        "잠금장치 길이조절(대)1P 민트",
        "이선내츄럴타올5p",
        "아기네일 깎기세트 민트",
        "미용타올 소10p",
        "쉐프온 망사수세미 20P",
        "간편 두배로청소기 S1SET",
        "식탁매트 보타니컬 가든 (매트4매+컵받침4매 세트) 1set",
        "카카오타올1p",
        "싱크대 물막이 그레이",
        "아트홈브라이텍스 스텐수세미 2P 40g",
        "줄무늬 쿠션 2m+코너 4p 베이지",
        "쉐프온(헤) 독일칼라행주 10매입",
        "빈티지 다용도 원형 우산꽂이 그물망",
        "코너보호대 동물 입체 2p 개구리",
        "잠금장치 길이조절 (중)1P 라벤더",
        "옻칠목기 그레이",
        "900메탈 4단선반(상) TJ-97",
        "간편밀대청소기 1p",
        "체크타올1p 1p",
        "쉐프온(헤) 엠보수세미 10매입",
        "전기모기채 그린",
        "수아르 하드 스퀘어도마 특대",
        "실리콘 그립 테이프 1M",
        "빈티지 다용도 우산꽂이 야생화",
        "항균밀대set (밀대+물걸레청소포) 1p",
        "티크우드 볶음스푼",
        "쉐프온 울스텐수세미 10P",
        "실리콘 그립 테이프 3M",
        "잠금장치 원터치 2P 그레이",
        "용기형 찜받침 종이호일 23cm 30매입",
        "산업용 900메탈 3단선반(하) TJ-99",
        "쿠션 2m+코너 4p 우드",
        "손끼임방지 펭귄 2P 화이트",
        "줄무늬 쿠션 2m+코너 4p 그레이",
        "전기모기채 블루",
        "위생 롤도마",
        "쉐프온(헤) 울스텐수세미 10매입",
        "원터치물병 720ml 화이트",
        "코너보호대 동물 입체 2p 곰돌이",
        "문받침이 발모양 1P 그레이",
        "베네치아타올2p",
        "미니쿠션 2m+코너 4p 핑크",
        "모서리 가드 네모 4P",
        "다용도 실리콘매트 브라운-나뭇잎",
        "쉐프온 은사수세미(중) 10P",
        "식탁매트 보타니컬 후르츠 (매트4매+컵받침4매 세트) 1set",
        "내츄럴타올1P",
        "초극세사 요술걸레1P",
        "450메탈5단선반(하) TJ-108",
        "다용도 실리콘매트 퍼플-티팟",
        "쉐프온 부직포수세미 10p",
        "아트홈브라이텍스 스텐수세미 1P 40g",
        "도모스텐원형텀블러(아이보리) 530ML",
        "코너보호대 부채 4P 화이트",
        "자연옻칠 타원수저",
        "잠금장치 원터치 2P 그린",
        "다용도 실리콘매트 핑크-나뭇잎",
        "이선내츄럴타올3p",
        "앤틱내츄럴타올2p",
        "산업용 900메탈 5단선반(하) TJ-101",
        "다용도 실리콘매트 핑크-티팟",
        "600메탈 4단선반(상) TJ-91",
        "내츄럴타올2p",
        "쿠션 2m+코너 4p 옐로우",
        "잠금장치 고래 1P 민트",
        "자연옻칠 궁중수저",
        "피첸시아타올1p",
        "자연옻칠 젓가락 2모",
        "간편 다용도청소기_리필2P S1SET",
        "퍼즐용기세트 일자 3p",
        "매직크리너 10p",
        "줄무늬 쿠션 2m+코너 4p 우드",
        "항균 극세사 리필 1p",
        "쉐프온(헤) 버블수세미 10매입",
        "간편샌들 네이비 250mm",
        "간편샌들 브라운 230mm",
        "넓은 쿠션 2m 핑크",
        "옻칠목기 크림",
        "도트욕실화(EVA) 1set",
        "베네치아타올1p",
        "독일행주7매입 7P",
        "아트홈브라이텍스 스텐수세미 3P 25g",
        "문닫힘방지 강아지 1P 블루",
        "코너보호대 동물 평면 4p 돼지",
        "간편샌들 네이비 280mm",
        "간편슈즈화이트 280",
        "원터치물병 1000ml 화이트",
        "간편 플라워청소기 S1SET",
        "산업용1200메탈 4단선반(하) TJ-103",
        "코너보호대 부채 4P 그린",
        "도모스텐원형머그(블랙) 330ML",
        "문콕 방지 4P",
        "순면 타올 3p",
        "식탁매트 컨트리 가든 1p",
        "퍼즐용기세트 항아리 3p",
        "문콕 방지 별원형 3P",
        "문받침이 발모양 1P 퍼플",
        "체크미용타올1p",
        "다용도 실리콘매트 차콜-티팟",
        "간편슈즈네이비 230",
        "다용도 실리콘매트 퍼플-이니셜H",
        "미니쿠션 2m+코너 4p 우드",
        "매직크리너 소1P",
        "도모스텐원형텀블러(블랙) 710ML",
        "빈티지 다용도 우산꽂이 도시",
        "잠금장치 고래 1P 화이트",
        "아기네일 깎기세트 핑크",
        "진공에어펌프&진공팩",
        "아트홈브라이텍스 간편 테이프크리너 리필 60매 벌크 12P",
        "도모스텐원형텀블러(스카이블루) 530ML",
        "도모스텐원형텀블러(블랙) 530ML",
        "간편먼지털이개 (특대)",
        "간편샌들 네이비 270mm",
        "잠금장치 길이조절 (중)1P 그린",
        "450메탈 3단선반(하) TJ-106",
        "아띠홈 오토디스펜서 액체형 360ml 화이트",
        "미용타올 중10p",
        "티크우드 에크팬",
        "아트 글라스 유리물병 1400ml flower1",
        "아트홈브라이텍스 필터수세미 1P",
        "다용도 실리콘매트 브라운-티팟",
        "피첸시아타올3p",
        "모서리가드 스마일원형 4P",
        "앤틱내츄럴타올1p",
        "수아르 하드 핸디 플레이팅 소",
        "매직크리너 중1P",
        "간편샌들 네이비 260mm",
        "쉐프온(헤) 요술행주 핑크 10매입",
        "진공팩set(10pcs)",
        "줄무늬 쿠션 2m+코너 4p 옐로우",
        "450메탈4단선반(하) TJ-107",
        "쉐프온 스텐수세미 10P",
        "티크우드 주걱",
        "물걸레 청소포 30매",
        "아트홈브라이텍스 스텐수세미 1P 50g",
        "잠금장치 길이조절(대)1P 핑크",
        "코너보호대 초승달 4P 민트",
        "티크우드 뒤집개(대)",
        "프림로즈 생리컵 세정제 150ml",
        "모서리가드 문양삼각 4P",
        "750메탈 5단선반(상) TJ-95",
        "간편슈즈화이트 250",
        "간편샌들 네이비 230mm",
        "식기건조대 1단 알뜰형",
        "앤틱내츄럴타올3p",
        "항균 샤워타올 황 옥 은1P",
        "다용도 실리콘매트 네이비-이니셜H",
        "순면 타올 2p",
        "티크우드 조리젓가락",
        "아트홈브라이텍스 간편 테이프크리너 리필 90매 벌크 12p",
        "순면행주 2p",
        "원터치물병 720ml 핑크",
        "다용도 실리콘매트 퍼플-나뭇잎",
        "손끼임방지 펭귄 2P 블루",
        "아트홈브라이텍스 간편 테이프크리너 리필 60매 3p",
        "독일칼라행주 3매입",
        "모서리 가드 원형 4P",
        "쉐프온(헤) 은사수세미 10매입",
        "도모스텐손잡이텀블러(블랙) 470ML",
        "순면 타올 1p",
        "문쿵잠금이 2P 민트",
        "다용도 매트 750메탈선반용 TJ-111",
        "모서리 가드 스마일세모 4P",
        "줄무늬 쿠션 2m+코너 4p 블루",
        "간편 타원청소기-리필 S2p",
        "매직크리너 특대1p",
        "간편슈즈화이트 270",
        "900메탈 3단선반(상) TJ-96",
        "코너보호대 초승달 4P 브라운",
        "초극세사 밀대 1p",
        "안전잠금이 2P 우드",
        "이선내츄럴타올2p",
        "미니쿠션 2m+코너 4p 브라운",
        "다용도 실리콘매트 민트-나뭇잎",
        "실리콘 그립 테이프 5M",
        "쉐프온(헤) 40스텐수세미 10매입",
        "간편슈즈화이트 240",
        "항균청소밀대 (초극세사+부직포) 1p",
        "아트홈브라이텍스 울스텐 수세미 1P",
        "문닫힘방지 강아지 1P 그린",
        "쉐프온(헤) 광수세미 10매입",
        "티크우드 뒤집개(중)",
        "물수건 20p",
        "다용도 매트 900메탈선반용 TJ-112",
        "간편 집게청소기_리필1P S1P",
        "내츄럴타올3p",
        "아트홈브라이텍스 스텐수세미 3P 40g",
        "수아르 하드 스퀘어도마 대",
        "아트 글라스 유리물병 1000ml flower1",
        "간편슈즈네이비 250",
        "간편슈즈네이비 260",
        "넓은 쿠션 2m 우드",
        "다용도 실리콘매트 민트-이니셜H",
        "매직크리너 홈-대1P",
        "쿠션 2m+코너 4p 핑크",
        "다기능 전기약탕기 3.0L",
        "쉐프온(헤) 은사수세미(중) 10매입",
        "간편 플러스청소기 1p",
        "다용도 매트 600메탈선반용 TJ-110",
        "간편슈즈네이비 280",
        "간편샌들 브라운 250mm",
        "간편슈즈네이비 240",
        "아트홈브라이텍스 스텐수세미 2P+1P 30g",
        "빈티지 다용도 원형 우산꽂이 삼각형",
        "쉐프온(헤) 망사수세미 10매입",
        "초극세사 요술행주1P",
        "넓은 쿠션 2m 브라운",
        "다용도 실리콘매트 차콜-나뭇잎",
        "아트 글라스 유리물병(M) 1400ml grapes",
        "문쿵잠금이 2P 퍼플",
        "다용도 실리콘매트 네이비-티팟",
        "순면칼라행주 3p",
        "빈티지 다용도 우산꽂이 우산",
        "아트홈브라이텍스 망사수세미 3P",
        "줄무늬 쿠션 2m+코너 4p 핑크",
        "다용도 매트 450메탈선반용 TJ-109",
        "도모스텐원형텀블러(스카이블루) 710ML",
        "아트홈브라이텍스 광수세미 2Pcs",
        "피첸시아타올2p",
        "아트홈브라이텍스 70스테인레스 수세미1P",
        "빈티지 다용도 우산꽂이 꽃",
        "빈티지 다용도 우산꽂이 트리",
        "아트홈브라이텍스 간편 테이프크리너 리필 90매 3p",
        "다용도 실리콘매트 핑크-스푼 포크",
        "간편샌들 브라운 260mm",
        "다용도 실리콘매트 브라운-스푼 포크",
        "다용도 실리콘매트 핑크-이니셜H",
        "미용타올 대10p",
        "다용도 실리콘매트 차콜-스푼 포크",
        "초극세사 밀대_리필 1p",
        "간편샌들 브라운 280mm",
        "수아르 하드 핸들스퀘어 중",
        "체크타올4P 4P",
        "다용도 실리콘매트 민트-티팟",
        "안전잠금이 2P 브라운",
        "아띠홈 핸드워시 리필350ml",
        "초극세사 (대) 밀대_리필 1p",
        "간편슈즈네이비 270",
        "600메탈 3단선반(상) TJ-90",
        "미니쿠션 2m+코너 4p 크림",
        "일회용 클린 마스크(MB필터) 50p",
        "전기모기채 핑크",
        "줄무늬 쿠션 2m+코너 4p 크림",
        "아트홈브라이텍스 부직포수세미 1P",
        "베네치아타올3p",
        "다용도 실리콘매트 퍼플-스푼 포크",
        "매직크리너 대1P",
        "도모스텐손잡이텀블러(아이보리) 590ML",
        "쿠션 2m+코너 4p 크림",
        "매직크리너 홈-중1P",
        "쉐프온독일행주 20p",
        "식탁매트 플라워 가든 1p",
        "베네치아타올4p",
        "플라워욕실화(EVA) 1set",
        "원터치물병 1000ml 핑크",
        "식탁매트 키즈ABCD 1p",
        "사각비누케이스 1P",
        "다용도 실리콘매트 네이비-나뭇잎",
        "아트홈브라이텍스 은사수세미 3P",
        "미니쿠션 2m+코너 4p 블루",
        "600메탈 5단선반(상) TJ-92",
        "900메탈 5단선반(상) TJ-98",
        "코너보호대 무당벌레 2p 그린",
        "손끼임방지 펭귄 2P 핑크",
        "빈티지 다용도 우산꽂이 빗방울",
        "아트홈브라이텍스 은사수세미(중) 1P",
        "간편 플라워청소기_리필 S1P",
        "체크미용타올4p",
        "다용도 실리콘매트 차콜-이니셜H",
        "줄무늬 쿠션 2m+코너 4p 브라운",
        "아트홈브라이텍스 은사수세미 1P",
        "다용도 실리콘매트 민트-스푼 포크",
        "식탁매트 해피홈 핑크 1p",
        "빈티지 다용도 우산꽂이 전화부스",
        "아트홈브라이텍스 울스텐 수세미 2P",
        "코너보호대 동물 입체 2p 고양이",
        "독일플라워행주 3p",
        "쿠션 2m+코너 4p 베이지",
        "쉐프온(헤) 순면행주 10매입",
        "쉐프온 은사수세미 10P",
        "독일행주ART 3p",
        "피첸시아타올4p",
        "간편먼지털이개 총채",
        "도모스텐원형텀블러(아이보리) 710ML",
        "코너보호대 동물 평면 4p 개구리",
        "간편 두배로청소기_리필 S1P",
        "도모스텐원형머그(아이보리) 330ML",
        "순면행주 20p",
        "아트홈브라이텍스 간편 테이프크리너 본품",
        "수아르 하드 스퀘어도마 중",
        "넓은 쿠션 2m 그레이",
        "코너보호대 초승달 4P 화이트",
        "넓은 쿠션 2m 베이지",
        "옻칠목기 브라운",
        "싱크대 물막이 화이트",
        "프림로즈 생리컵 대형",
        "750메탈 3단선반(상) TJ-93",
        "간편 다용도청소기 S1SET",
        "간편샌들 브라운 240mm",
        "다용도 실리콘매트 네이비-스푼 포크",
        "보르네오 손잡이 롱플레이트도마",
        "체크타올 중10p",
        "다용도 실리콘매트 브라운-이니셜H",
        "아트 글라스 유리물병 1400ml flower2",
        "아트 글라스 유리물병(M) 1400ml flower2",
        "식기건조대 2단",
        "간편슈즈화이트260",
        "간편 타원청소기 S1P",
        "쿠션 2m+코너 4p 그레이",
        "보르네오 원형도마",
        "티크우드 수저세트",
        "항균핸디밀대 1p",
        "테잎쿠션 2M 1P",
        "간편 플러스청소기 리필 1P",
        "간편슈즈화이트 230",
        "에어프라이어 종이호일 23cm 30매",
        "카카오타올4p",
        "퍼즐욕실화(EVA) 트임1set",
        "쉐프온(헤) 요술행주 블루 10매입",
        "보르네오 핸들 스퀘어도마(대)",
        "이선내츄럴타올1p",
        "파스텔행주 2p",
        "쿠션 2m+코너 4p 브라운",
        "잠금장치 길이조절 (중)1P 그레이",
        "아트홈브라이텍스 부직포수세미 3P",
        "프림로즈 생리컵 소형",
        "항균 부직포(정전기)리필 10p",
        "코너보호대 무당벌레 2p 블루",
        "아트홈브라이텍스 필터수세미 2P",
        "퍼즐욕실화(EVA) 막힘1set",
        "구름비누케이스 1P",
        "간편샌들 네이비 240mm",
        "아트홈브라이텍스 망사수세미 2P",
        "도모스텐손잡이텀블러(아이보리) 470ML",
        "아띠홈 오토디스펜서 거품형 400ml 화이트",
        "넓은 쿠션 2m 옐로우"
    ]
    call(product_names)