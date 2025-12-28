#!/usr/bin/env python3
"""
저작물 이미지 유사도 분석 시스템
CLIP/VLM + 구조적 특징 추출 하이브리드 모델

CLIP/VLM의 한계를 보완하기 위해:
1. Perceptual Hashing (pHash, dHash) - 구조적 유사도
2. SSIM (Structural Similarity Index) - 픽셀 레벨 구조 비교
3. Deep Features (ResNet/EfficientNet) - 특징 벡터 비교
"""
import hashlib
import numpy as np
from typing import Tuple, Dict, List, Optional
from PIL import Image
import cv2
from scipy.spatial.distance import cosine
from dataclasses import dataclass


@dataclass
class SimilarityScore:
    """유사도 점수 결과"""
    overall_score: float  # 0.0 ~ 1.0
    perceptual_hash_score: float
    ssim_score: float
    deep_feature_score: float
    semantic_score: float  # CLIP/VLM score
    confidence: float
    method_weights: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            "overall_score": round(self.overall_score, 4),
            "breakdown": {
                "perceptual_hash": round(self.perceptual_hash_score, 4),
                "structural_similarity": round(self.ssim_score, 4),
                "deep_features": round(self.deep_feature_score, 4),
                "semantic_similarity": round(self.semantic_score, 4)
            },
            "confidence": round(self.confidence, 4),
            "method_weights": self.method_weights,
            "interpretation": self._interpret_score()
        }
    
    def _interpret_score(self) -> str:
        """점수 해석"""
        if self.overall_score >= 0.95:
            return "거의 동일 (Almost Identical)"
        elif self.overall_score >= 0.85:
            return "매우 유사 (Very Similar)"
        elif self.overall_score >= 0.70:
            return "유사 (Similar)"
        elif self.overall_score >= 0.50:
            return "부분 유사 (Partially Similar)"
        else:
            return "상이함 (Different)"


class ImageSimilarityAnalyzer:
    """이미지 유사도 종합 분석 시스템"""
    
    def __init__(
        self,
        use_perceptual_hash: bool = True,
        use_ssim: bool = True,
        use_deep_features: bool = True,
        use_semantic: bool = True
    ):
        """
        초기화
        
        Args:
            use_perceptual_hash: Perceptual hashing 사용 여부
            use_ssim: SSIM 사용 여부
            use_deep_features: Deep feature extraction 사용 여부
            use_semantic: Semantic similarity (CLIP/VLM) 사용 여부
        """
        self.use_perceptual_hash = use_perceptual_hash
        self.use_ssim = use_ssim
        self.use_deep_features = use_deep_features
        self.use_semantic = use_semantic
        
        # 방법별 가중치 (조정 가능)
        self.weights = {
            "perceptual_hash": 0.25,
            "ssim": 0.25,
            "deep_features": 0.30,
            "semantic": 0.20
        }
        
        # Deep learning 모델은 lazy loading
        self._feature_extractor = None
    
    def analyze_similarity(
        self,
        image1_path: str,
        image2_path: str,
        semantic_score: Optional[float] = None
    ) -> SimilarityScore:
        """
        두 이미지의 유사도 종합 분석
        
        Args:
            image1_path: 첫 번째 이미지 경로
            image2_path: 두 번째 이미지 경로
            semantic_score: CLIP/VLM에서 계산된 의미적 유사도 (optional)
            
        Returns:
            SimilarityScore 객체
        """
        # 이미지 로드
        img1 = Image.open(image1_path).convert("RGB")
        img2 = Image.open(image2_path).convert("RGB")
        
        scores = {}
        active_weights = {}
        
        # 1. Perceptual Hashing
        if self.use_perceptual_hash:
            phash_score = self._calculate_perceptual_hash_similarity(img1, img2)
            scores["perceptual_hash"] = phash_score
            active_weights["perceptual_hash"] = self.weights["perceptual_hash"]
        else:
            scores["perceptual_hash"] = 0.0
        
        # 2. SSIM
        if self.use_ssim:
            ssim_score = self._calculate_ssim(img1, img2)
            scores["ssim"] = ssim_score
            active_weights["ssim"] = self.weights["ssim"]
        else:
            scores["ssim"] = 0.0
        
        # 3. Deep Features
        if self.use_deep_features:
            deep_score = self._calculate_deep_feature_similarity(img1, img2)
            scores["deep_features"] = deep_score
            active_weights["deep_features"] = self.weights["deep_features"]
        else:
            scores["deep_features"] = 0.0
        
        # 4. Semantic (CLIP/VLM)
        if self.use_semantic and semantic_score is not None:
            scores["semantic"] = semantic_score
            active_weights["semantic"] = self.weights["semantic"]
        else:
            scores["semantic"] = 0.0
        
        # 가중치 정규화
        total_weight = sum(active_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in active_weights.items()}
        else:
            normalized_weights = active_weights
        
        # 가중 평균 계산
        overall_score = sum(
            scores[method] * normalized_weights.get(method, 0.0)
            for method in scores.keys()
        )
        
        # Confidence 계산 (점수들의 분산이 작을수록 높은 confidence)
        score_variance = np.var(list(scores.values()))
        confidence = 1.0 / (1.0 + score_variance)
        
        return SimilarityScore(
            overall_score=overall_score,
            perceptual_hash_score=scores["perceptual_hash"],
            ssim_score=scores["ssim"],
            deep_feature_score=scores["deep_features"],
            semantic_score=scores["semantic"],
            confidence=confidence,
            method_weights=normalized_weights
        )
    
    # ==================== Perceptual Hashing ====================
    
    def _calculate_perceptual_hash_similarity(
        self,
        img1: Image.Image,
        img2: Image.Image,
        hash_size: int = 8
    ) -> float:
        """
        Perceptual Hash (pHash) 기반 유사도 계산
        
        Args:
            img1, img2: PIL Image 객체
            hash_size: 해시 크기 (기본 8x8 = 64bit)
            
        Returns:
            0.0 ~ 1.0 유사도 점수
        """
        # Average Hash (aHash) 계산
        ahash1 = self._average_hash(img1, hash_size)
        ahash2 = self._average_hash(img2, hash_size)
        
        # Difference Hash (dHash) 계산
        dhash1 = self._difference_hash(img1, hash_size)
        dhash2 = self._difference_hash(img2, hash_size)
        
        # Hamming distance 계산
        ahash_sim = 1.0 - (self._hamming_distance(ahash1, ahash2) / (hash_size * hash_size))
        dhash_sim = 1.0 - (self._hamming_distance(dhash1, dhash2) / (hash_size * hash_size))
        
        # 두 방법의 평균
        return (ahash_sim + dhash_sim) / 2.0
    
    def _average_hash(self, img: Image.Image, hash_size: int = 8) -> str:
        """Average Hash 계산"""
        # 리사이즈 및 그레이스케일 변환
        img_resized = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS).convert("L")
        pixels = np.array(img_resized)
        
        # 평균값 계산
        avg = pixels.mean()
        
        # 이진 해시 생성
        hash_bits = (pixels > avg).flatten()
        return ''.join('1' if bit else '0' for bit in hash_bits)
    
    def _difference_hash(self, img: Image.Image, hash_size: int = 8) -> str:
        """Difference Hash 계산"""
        # 리사이즈 (hash_size+1 너비)
        img_resized = img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS).convert("L")
        pixels = np.array(img_resized)
        
        # 인접 픽셀 차이 계산
        diff = pixels[:, 1:] > pixels[:, :-1]
        
        return ''.join('1' if bit else '0' for bit in diff.flatten())
    
    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """Hamming distance 계산"""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    # ==================== SSIM ====================
    
    def _calculate_ssim(
        self,
        img1: Image.Image,
        img2: Image.Image,
        window_size: int = 11
    ) -> float:
        """
        Structural Similarity Index (SSIM) 계산
        
        Args:
            img1, img2: PIL Image 객체
            window_size: SSIM 윈도우 크기
            
        Returns:
            0.0 ~ 1.0 SSIM 점수
        """
        try:
            # 동일한 크기로 리사이즈
            size = (256, 256)
            img1_resized = img1.resize(size, Image.Resampling.LANCZOS)
            img2_resized = img2.resize(size, Image.Resampling.LANCZOS)
            
            # NumPy 배열로 변환
            arr1 = np.array(img1_resized)
            arr2 = np.array(img2_resized)
            
            # 그레이스케일 변환
            if len(arr1.shape) == 3:
                arr1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2GRAY)
            if len(arr2.shape) == 3:
                arr2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY)
            
            # SSIM 계산
            try:
                from skimage.metrics import structural_similarity as ssim
                score = ssim(arr1, arr2, win_size=window_size, data_range=255)
            except ImportError:
                # skimage 없으면 간단한 MSE 기반 유사도 사용
                score = self._simple_ssim_fallback(arr1, arr2)
            
            return float(score)
        
        except Exception as e:
            print(f"SSIM calculation error: {e}")
            return 0.5
    
    def _simple_ssim_fallback(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """SSIM fallback (MSE 기반 간단한 구조 유사도)"""
        # Mean Squared Error 계산
        mse = np.mean((arr1.astype(float) - arr2.astype(float)) ** 2)
        
        # MSE를 유사도로 변환 (0~1 범위)
        max_mse = 255 ** 2
        similarity = 1.0 - (mse / max_mse)
        
        return float(max(0.0, min(1.0, similarity)))
    
    # ==================== Deep Features ====================
    
    def _calculate_deep_feature_similarity(
        self,
        img1: Image.Image,
        img2: Image.Image
    ) -> float:
        """
        Deep Learning 특징 벡터 기반 유사도 계산
        
        Args:
            img1, img2: PIL Image 객체
            
        Returns:
            0.0 ~ 1.0 유사도 점수 (cosine similarity)
        """
        # Feature extractor lazy loading
        if self._feature_extractor is None:
            self._feature_extractor = self._load_feature_extractor()
        
        # 특징 추출
        features1 = self._extract_features(img1)
        features2 = self._extract_features(img2)
        
        # Cosine similarity 계산
        similarity = 1.0 - cosine(features1, features2)
        
        return float(max(0.0, min(1.0, similarity)))
    
    def _load_feature_extractor(self):
        """특징 추출 모델 로드 (간단한 버전)"""
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms
            
            # ResNet50 pretrained 모델
            model = models.resnet50(pretrained=True)
            # 마지막 FC layer 제거 (특징 벡터만 추출)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model.eval()
            
            # 전처리 transform
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            return {"model": model, "transform": transform, "device": "cpu"}
        
        except ImportError:
            # PyTorch 없으면 간단한 대체 방법 사용
            return None
    
    def _extract_features(self, img: Image.Image) -> np.ndarray:
        """이미지 특징 벡터 추출"""
        if self._feature_extractor is None:
            # PyTorch 없으면 간단한 color histogram 사용
            return self._extract_color_histogram(img)
        
        try:
            import torch
            
            model = self._feature_extractor["model"]
            transform = self._feature_extractor["transform"]
            
            # 이미지 전처리
            img_tensor = transform(img).unsqueeze(0)
            
            # 특징 추출
            with torch.no_grad():
                features = model(img_tensor)
            
            # Flatten
            features = features.squeeze().numpy()
            
            return features
        
        except Exception:
            # 에러 시 fallback
            return self._extract_color_histogram(img)
    
    def _extract_color_histogram(self, img: Image.Image, bins: int = 32) -> np.ndarray:
        """간단한 Color Histogram 특징 (fallback)"""
        img_resized = img.resize((256, 256), Image.Resampling.LANCZOS)
        arr = np.array(img_resized)
        
        if len(arr.shape) == 2:  # Grayscale
            hist, _ = np.histogram(arr, bins=bins, range=(0, 256))
        else:  # RGB
            hist_r, _ = np.histogram(arr[:, :, 0], bins=bins, range=(0, 256))
            hist_g, _ = np.histogram(arr[:, :, 1], bins=bins, range=(0, 256))
            hist_b, _ = np.histogram(arr[:, :, 2], bins=bins, range=(0, 256))
            hist = np.concatenate([hist_r, hist_g, hist_b])
        
        # 정규화
        hist = hist.astype(float)
        hist = hist / (hist.sum() + 1e-7)
        
        return hist


def batch_similarity_analysis(
    image_pairs: List[Tuple[str, str]],
    semantic_scores: Optional[List[float]] = None
) -> List[SimilarityScore]:
    """
    여러 이미지 쌍에 대한 일괄 유사도 분석
    
    Args:
        image_pairs: (이미지1 경로, 이미지2 경로) 튜플 리스트
        semantic_scores: CLIP/VLM 점수 리스트 (optional)
        
    Returns:
        SimilarityScore 객체 리스트
    """
    analyzer = ImageSimilarityAnalyzer()
    results = []
    
    for i, (img1_path, img2_path) in enumerate(image_pairs):
        semantic_score = semantic_scores[i] if semantic_scores else None
        
        try:
            score = analyzer.analyze_similarity(img1_path, img2_path, semantic_score)
            results.append(score)
        except Exception as e:
            print(f"Error analyzing {img1_path} vs {img2_path}: {e}")
            results.append(None)
    
    return results


if __name__ == "__main__":
    # 테스트
    import tempfile
    import os
    
    print("=== 이미지 유사도 분석 시스템 테스트 ===\n")
    
    # Dummy test (실제 이미지 없이 기능 검증)
    analyzer = ImageSimilarityAnalyzer()
    
    # 간단한 테스트 이미지 생성
    from PIL import Image, ImageDraw
    
    # 임시 디렉토리 사용
    temp_dir = tempfile.gettempdir()
    
    # 이미지 1: 빨간 원
    img1 = Image.new("RGB", (100, 100), "white")
    draw1 = ImageDraw.Draw(img1)
    draw1.ellipse([20, 20, 80, 80], fill="red")
    
    # 이미지 2: 빨간 원 (약간 다른 크기)
    img2 = Image.new("RGB", (100, 100), "white")
    draw2 = ImageDraw.Draw(img2)
    draw2.ellipse([25, 25, 75, 75], fill="red")
    
    # 이미지 3: 파란 사각형 (완전히 다름)
    img3 = Image.new("RGB", (100, 100), "white")
    draw3 = ImageDraw.Draw(img3)
    draw3.rectangle([20, 20, 80, 80], fill="blue")
    
    # 임시 저장
    img1_path = os.path.join(temp_dir, "test_img1.png")
    img2_path = os.path.join(temp_dir, "test_img2.png")
    img3_path = os.path.join(temp_dir, "test_img3.png")
    
    img1.save(img1_path)
    img2.save(img2_path)
    img3.save(img3_path)
    
    # 유사한 이미지 비교
    print("1. 유사한 이미지 비교 (빨간 원 vs 약간 작은 빨간 원)")
    score1 = analyzer.analyze_similarity(img1_path, img2_path, semantic_score=0.92)
    print(f"   Overall Score: {score1.overall_score:.2%}")
    print(f"   Details: {score1.to_dict()}\n")
    
    # 다른 이미지 비교
    print("2. 다른 이미지 비교 (빨간 원 vs 파란 사각형)")
    score2 = analyzer.analyze_similarity(img1_path, img3_path, semantic_score=0.35)
    print(f"   Overall Score: {score2.overall_score:.2%}")
    print(f"   Details: {score2.to_dict()}\n")
    
    # 정리
    os.remove(img1_path)
    os.remove(img2_path)
    os.remove(img3_path)
    
    print("=== 테스트 완료 ===")
