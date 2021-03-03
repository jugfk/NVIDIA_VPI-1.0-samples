# 2D 이미지 컨볼 루션

2D 이미지 컨볼 루션 애플리케이션은 입력 이미지의 가장자리가있는 이미지를 출력하여 결과를 edge.png에 저장합니다 . 사용자는 처리에 사용할 백엔드를 정의 할 수 있습니다.

노트

컨볼 루션은 현재 단일 채널 이미지에만 지원되므로 출력은 회색조로 표시됩니다.
이 샘플은 다음을 보여줍니다.

VPI 스트림 생성 및 삭제.
VPI와 함께 사용할 OpenCV cv :: Mat 이미지 래핑.
출력이 기록 될 VPI 관리 2D 이미지 만들기.
Convolve2D 알고리즘을 만들고 호출하여 사용자 지정 커널을 전달합니다.
간단한 스트림 동기화.
CPU 측에서 콘텐츠에 액세스하기위한 이미지 잠금.
오류 처리.
환경 정리.
명령
사용법은 다음과 같습니다.

```./vpi_sample_01_convolve_2d <백엔드> <입력 이미지>```

백엔드 : cpu , cuda 또는 pva ; 처리를 수행 할 백엔드를 정의합니다.

입력 이미지 : 입력 이미지 파일 이름; png, jpeg 및 기타 가능합니다.
다음은 한 가지 예입니다.

```./vpi_sample_01_convolve_2d cpu ../assets/kodim8.png```

이것은 CUDA 백엔드와 제공된 샘플 이미지 중 하나를 사용하고 있습니다. 알고리즘에 의해 부과 된 제약을 고려하여 다른 이미지로 시도 할 수 있습니다.

# 결과

입력 이미지

![입력이미지](https://raw.githubusercontent.com/jugfk/NVIDIA_VPI-1.0-samples/main/assets/kodim08.png)

출력 이미지

![출력이미지](https://raw.githubusercontent.com/jugfk/NVIDIA_VPI-1.0-samples/main/Images/edges_cpu.png)

