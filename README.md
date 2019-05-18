# WATCHME Background App
Main C# 애플리케이션에서 background process로 실행되는 Python 스크립트입니다.

## Install
패키징을 안했으니 그냥 스크립트 채로 돌리시면 됩니다.

최신 버전의 [Python3](https://www.python.org/downloads/)을 설치하고(이제는 `pip`도 딸려옴), `requirements.txt`가 있는 폴더에서 `pip install -r requirements.txt`를 실행하세요.

~~사실 저는 맥이라 잘 모르겠어요. 잘 되니 재성아?~~

아무튼 삽질 조금만 하면 깔릴 겁니다.

## Docs

### Mode

- `python main.py` 또는 `python main.py 0`: C# 애플리케이션에서 프로세스를 생성할 때는 이렇게 해주세요; 디버깅을 위한 다른 출력이나 `imshow` 등을 제외하고 (keras가 로딩된 뒤에) 전달되는 json이 한 줄씩 출력됩니다.
- `python main.py 1`: 디버깅 모드; 현재 EAR 등의 출력과 현재 화면이 출력됩니다.
- `python main.py 2`: **자동 EAR 커스터마이제이션**을 할 때 이걸 실행해주세요! 이건 내가 봐도 좀 오졌는데 나중에 설명해 드릴게요.
 
### Spec

```json
{"closed": -1, "stare": -1}
{"closed": 0, "stare": 4}
{"closed": 0, "stare": 0}
{"closed": 0, "stare": 1}
{"closed": 0, "stare": 4}
```
> 이런 식으로 한 줄씩 나와요.

- `closed`: 현재 사용자의 눈이 감겼는가?
  - 감겼다면 `1`, 감기지 않았다면 `0`

요건 제 경험상 (20개까지 무시하고) 50번 이상 연속으로 1이 날아오면 조는 걸로 해주면 될 것 같아요!

이걸 이용해서 일정 오프셋 동안 눈이 감겨 있으면 뭘 처리하는 식으로도 할 수 있어요.

- `stare`: 사용자가 보는 화면의 위치 
  - `0`이면 **왼쪽 아래**(bottom_left)
  - `1`이면 **오른쪽 아래**(bottom_right)
  - `2`이면 **중앙**(normal)
  - `3`이면 **왼쪽 위**(top_left)
  - `4`이면 **오른쪽 위**(top_right)

오른쪽 아래 보면 시계 띄워주고 하는 건데 이것도 오차를 감안해서 해줘야 할 것 같아요.

일단 모델 예측 결과가 반영되는 걸 조금씩 다르게 해서 고쳐보도록 노력은 해볼게요.

## Refs
다른 레포들 >< ~~404 뜨면 아직 비공개한 거예요!~~

- [watchme-closed-eye-detection](https://github.com/junhoyeo/watchme-closed-eye-detection): 눈 감았나 안 감았나 보는 거(이거 코드 대부분을 고쳐서 여기로 넣긴 함)
- [watchme-ai-pupil-tracker](https://github.com/junhoyeo/watchme-ai-pupil-tracker): model 데이터셋이랑 빌드하는 거 있음
