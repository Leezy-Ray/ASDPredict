"""
TwoTST API服务器
Flask后端服务，提供fMRI数据预测和异常连接分析接口
"""

import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback

# API根目录
API_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(API_ROOT)  # 项目根目录
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')  # 数据目录
sys.path.insert(0, API_ROOT)

from utils.data_processor import process_input_data
from services.prediction_service import PredictionService
from services.connection_analysis_service import ConnectionAnalysisService
from services.sample_service import get_sample_service

# 配置（使用 data 目录内的 checkpoints）
DEFAULT_CHECKPOINT = os.path.join(DATA_ROOT, 'checkpoints', 'best_model.pt')
DEFAULT_RESULTS_JSON = os.path.join(DATA_ROOT, 'checkpoints', 'results.json')
UPLOAD_FOLDER = os.path.join(API_ROOT, 'uploads')
ALLOWED_EXTENSIONS = {'csv', 'json'}

# 创建Flask应用
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# 启用CORS，允许前端跨域访问
CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/predict": {"origins": "*"},
    r"/analyze": {"origins": "*"},
    r"/health": {"origins": "*"},
    r"/cc200/*": {"origins": "*"}
})

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 全局服务实例（延迟加载）
prediction_service = None
connection_analysis_service = ConnectionAnalysisService(n_rois=200)


def get_prediction_service():
    """获取预测服务实例（单例）"""
    global prediction_service
    if prediction_service is None:
        checkpoint = request.args.get('checkpoint', DEFAULT_CHECKPOINT)
        results_json = request.args.get('results_json', DEFAULT_RESULTS_JSON)
        device = request.args.get('device', 'cuda')
        prediction_service = PredictionService(checkpoint, results_json, device)
    return prediction_service


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'service': 'TwoTST API Server',
        'version': '1.0.0'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    预测接口
    
    接收fMRI数据（JSON或CSV），返回每个窗口的预测概率
    
    请求格式:
    1. JSON Body: {"timeseries": [[...], [...]]} 或 {"data": [[...], [...]]}
    2. CSV文件上传: Content-Type: multipart/form-data
    
    查询参数:
    - window_size: 滑动窗口大小（默认32）
    - stride: 滑动窗口步长（默认16）
    - batch_size: 批次大小（默认32）
    - align_cc200: 是否对齐到CC200（默认true）
    - return_attention: 是否返回注意力权重（默认false）
    
    返回:
    {
        "window_predictions": [
            {
                "window_index": 0,
                "asd_probability": 0.85,
                "prediction": 1,
                "logits": [0.5, 1.2]
            },
            ...
        ],
        "summary": {
            "total_windows": 10,
            "mean_asd_probability": 0.78,
            ...
        }
    }
    """
    try:
        # 获取参数
        window_size = int(request.args.get('window_size', 32))
        stride = int(request.args.get('stride', 16))
        batch_size = int(request.args.get('batch_size', 32))
        align_cc200 = request.args.get('align_cc200', 'true').lower() == 'true'
        return_attention = request.args.get('return_attention', 'false').lower() == 'true'
        
        # 处理输入数据
        data = None
        input_format = 'auto'
        
        if 'file' in request.files:
            # 文件上传
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': '没有选择文件'}), 400
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                if filename.endswith('.csv'):
                    data = filepath
                    input_format = 'csv'
                elif filename.endswith('.json'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        import json
                        data = json.load(f)
                    input_format = 'json'
                    os.remove(filepath)  # 清理临时文件
        elif request.is_json:
            # JSON body
            data = request.get_json()
            input_format = 'json'
        elif request.content_type and 'application/json' in request.content_type:
            # JSON字符串
            data = request.get_data(as_text=True)
            input_format = 'json'
        else:
            return jsonify({'error': '请提供JSON数据或上传CSV/JSON文件'}), 400
        
        # 处理数据
        processed_data = process_input_data(
            data=data,
            input_format=input_format,
            window_size=window_size,
            stride=stride,
            align_cc200=align_cc200
        )
        
        # 预测
        pred_service = get_prediction_service()
        predictions = pred_service.predict(
            processed_data['windowed_data'],
            batch_size=batch_size,
            return_attention=return_attention
        )
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'data_info': {
                'n_rois': processed_data['n_rois'],
                'n_timepoints': processed_data['n_timepoints'],
                'n_windows': len(processed_data['windowed_data']['timeseries'])
            }
        })
    
    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error in /predict: {error_msg}\n{traceback_str}")
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': traceback_str if app.debug else None
        }), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    异常连接分析接口
    
    接收fMRI数据，返回预测结果和异常连接分析
    
    请求格式: 同 /predict
    
    查询参数: 同 /predict，额外：
    - threshold: ASD概率阈值（默认0.5）
    - top_k: 返回top-k异常连接（默认50）
    
    返回:
    {
        "predictions": {...},  # 同 /predict 返回
        "abnormal_connections": {
            "abnormal_connections": [
                {
                    "roi_i": 10,
                    "roi_j": 45,
                    "difference": 0.15,
                    "asd_connection_strength": 0.8,
                    "tc_connection_strength": 0.65
                },
                ...
            ],
            "connection_statistics": {...},
            "summary": {...}
        }
    }
    """
    try:
        # 获取参数
        window_size = int(request.args.get('window_size', 32))
        stride = int(request.args.get('stride', 16))
        batch_size = int(request.args.get('batch_size', 32))
        align_cc200 = request.args.get('align_cc200', 'true').lower() == 'true'
        threshold = float(request.args.get('threshold', 0.5))
        top_k = int(request.args.get('top_k', 50))
        
        # 处理输入数据（同predict）
        data = None
        input_format = 'auto'
        
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': '没有选择文件'}), 400
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                if filename.endswith('.csv'):
                    data = filepath
                    input_format = 'csv'
                elif filename.endswith('.json'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        import json
                        data = json.load(f)
                    input_format = 'json'
                    os.remove(filepath)
        elif request.is_json:
            data = request.get_json()
            input_format = 'json'
        elif request.content_type and 'application/json' in request.content_type:
            data = request.get_data(as_text=True)
            input_format = 'json'
        else:
            return jsonify({'error': '请提供JSON数据或上传CSV/JSON文件'}), 400
        
        # 处理数据
        processed_data = process_input_data(
            data=data,
            input_format=input_format,
            window_size=window_size,
            stride=stride,
            align_cc200=align_cc200
        )
        
        # 预测
        pred_service = get_prediction_service()
        predictions = pred_service.predict(
            processed_data['windowed_data'],
            batch_size=batch_size,
            return_attention=False
        )
        
        # 异常连接分析
        abnormal_connections = connection_analysis_service.analyze_abnormal_connections(
            pcc_vectors=processed_data['windowed_data']['pcc_vectors'],
            predictions=predictions,
            threshold=threshold,
            top_k=top_k
        )
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'abnormal_connections': abnormal_connections,
            'data_info': {
                'n_rois': processed_data['n_rois'],
                'n_timepoints': processed_data['n_timepoints'],
                'n_windows': len(processed_data['windowed_data']['timeseries'])
            }
        })
    
    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error in /analyze: {error_msg}\n{traceback_str}")
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': traceback_str if app.debug else None
        }), 500


@app.route('/cc200/rois', methods=['GET'])
def get_cc200_rois():
    """获取CC200的ROI信息"""
    roi_labels = connection_analysis_service.get_cc200_roi_labels()
    return jsonify({
        'success': True,
        'rois': roi_labels,
        'total_rois': len(roi_labels)
    })


@app.route('/api/samples', methods=['GET'])
def get_samples():
    """
    获取样本列表接口
    
    查询参数:
    - n_asd: ASD样本数量（默认5）
    - n_control: 正常对照组样本数量（默认5）
    
    返回:
    {
        "success": true,
        "samples": [
            {
                "id": 0,
                "original_id": 123,
                "type": "asd",
                "label": 1,
                "name": "ASD-1",
                "description": "自闭症样本 #1"
            },
            ...
        ]
    }
    """
    try:
        n_asd = int(request.args.get('n_asd', 5))
        n_control = int(request.args.get('n_control', 5))
        
        sample_service = get_sample_service()
        samples = sample_service.get_samples(n_asd=n_asd, n_control=n_control)
        
        # #region agent log
        import json as json_module
        log_data = {
            "location": "app.py:get_samples",
            "message": "Returning samples to frontend",
            "data": {
                "samples_count": len(samples),
                "first_sample": samples[0] if samples else None,
                "sample_names": [s.get('name') for s in samples[:3]]
            },
            "timestamp": int(__import__('time').time() * 1000),
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "H"
        }
        try:
            with open(r'd:\workplace\ASDModelPredict\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json_module.dumps(log_data, ensure_ascii=False) + '\n')
        except: pass
        # #endregion
        
        return jsonify({
            'success': True,
            'samples': samples,
            'total': len(samples)
        })
    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error in /api/samples: {error_msg}\n{traceback_str}")
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': traceback_str if app.debug else None
        }), 500


@app.route('/api/analyze-sample', methods=['POST'])
def analyze_sample():
    """
    分析选定样本的接口
    
    请求体:
    {
        "sample_id": 0,  # 样本ID（从/api/samples获取）
        "window_size": 32,  # 可选，窗口大小
        "stride": 16,  # 可选，滑动步长
        "threshold": 0.5,  # 可选，ASD概率阈值
        "top_k": 50  # 可选，返回top-k异常连接
    }
    
    返回:
    {
        "success": true,
        "predictions": {...},  # 窗口预测结果
        "abnormal_connections": {...},  # 异常连接分析
        "data_info": {...}
    }
    """
    try:
        # 获取参数
        data = request.get_json()
        if not data or 'sample_id' not in data:
            return jsonify({'error': '请提供sample_id'}), 400
        
        sample_id = int(data['sample_id'])
        window_size = int(data.get('window_size', 32))
        stride = int(data.get('stride', 16))
        threshold = float(data.get('threshold', 0.5))
        top_k = int(data.get('top_k', 50))
        batch_size = int(data.get('batch_size', 32))
        
        # 获取样本数据
        sample_service = get_sample_service()
        sample_data = sample_service.get_sample_data(sample_id)
        
        if sample_data is None:
            return jsonify({'error': f'找不到样本ID {sample_id}'}), 404
        
        # 确保数据格式正确 (n_timepoints, n_rois)
        if sample_data.ndim != 2:
            return jsonify({'error': f'数据维度错误: 期望2D (n_timepoints, n_rois)，实际{sample_data.ndim}D'}), 400
        
        print(f"Sample data shape before processing: {sample_data.shape}")
        
        # 处理数据
        processed_data = process_input_data(
            data=sample_data,
            input_format='array',
            window_size=window_size,
            stride=stride,
            align_cc200=True
        )
        
        print(f"Processed data - n_rois: {processed_data['n_rois']}, n_timepoints: {processed_data['n_timepoints']}")
        print(f"Windowed data - pcc_vectors shape: {processed_data['windowed_data']['pcc_vectors'].shape}")
        
        # 预测
        pred_service = get_prediction_service()
        predictions = pred_service.predict(
            processed_data['windowed_data'],
            batch_size=batch_size,
            return_attention=False
        )
        
        # 异常连接分析
        abnormal_connections = connection_analysis_service.analyze_abnormal_connections(
            pcc_vectors=processed_data['windowed_data']['pcc_vectors'],
            predictions=predictions,
            threshold=threshold,
            top_k=top_k
        )
        
        # 转换异常连接格式以匹配前端期望
        # 前端期望格式：region1, region2, type, strength
        formatted_connections = []
        for conn in abnormal_connections['abnormal_connections']:
            # 判断是过度连接还是连接不足
            asd_strength = conn['asd_connection_strength']
            tc_strength = conn['tc_connection_strength']
            diff = asd_strength - tc_strength
            
            formatted_connections.append({
                'region1': conn['roi_i'],
                'region2': conn['roi_j'],
                'type': 'hyper' if diff > 0 else 'hypo',
                'strength': abs(conn['difference']),
                'asd_strength': asd_strength,
                'tc_strength': tc_strength,
                'difference': conn['difference']
            })
        
        # 构建响应（不返回 fmri_data，前端不需要）
        response_data = {
            'success': True,
            'predictions': predictions,
            'abnormal_connections': {
                'connections': formatted_connections,
                'statistics': abnormal_connections['connection_statistics'],
                'summary': abnormal_connections['summary']
            },
            'data_info': {
                'n_rois': processed_data['n_rois'],
                'n_timepoints': processed_data['n_timepoints'],
                'n_windows': len(processed_data['windowed_data']['timeseries']),
                'sample_id': sample_id
            }
        }
        
        print(f"Response summary:")
        print(f"  - Windows: {len(predictions.get('window_predictions', []))}")
        print(f"  - Abnormal connections: {len(formatted_connections)}")
        print(f"  - Sample probabilities: {[w['asd_probability'] for w in predictions.get('window_predictions', [])[:3]]}")
        
        return jsonify(response_data)
    
    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error in /api/analyze-sample: {error_msg}\n{traceback_str}")
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': traceback_str if app.debug else None
        }), 500


@app.route('/api/sample-data', methods=['GET'])
def get_sample_data():
    """
    获取样本数据接口（不进行分析）
    
    查询参数:
    - sample_id: 样本ID（从/api/samples获取）
    
    返回:
    {
        "success": true,
        "data": [[...], [...]],  # fMRI时间序列数据 (n_timepoints, n_rois)
        "data_info": {
            "n_rois": 200,
            "n_timepoints": 200
        }
    }
    """
    try:
        sample_id_str = request.args.get('sample_id')
        if sample_id_str is None:
            return jsonify({'error': '请提供sample_id'}), 400
        
        sample_id = int(sample_id_str)
        
        # 获取样本数据
        sample_service = get_sample_service()
        sample_data = sample_service.get_sample_data(sample_id)
        
        if sample_data is None:
            return jsonify({'error': f'找不到样本ID {sample_id}'}), 404
        
        # 确保数据格式正确 (n_timepoints, n_rois)
        if sample_data.ndim != 2:
            return jsonify({'error': f'数据维度错误: 期望2D (n_timepoints, n_rois)，实际{sample_data.ndim}D'}), 400
        
        return jsonify({
            'success': True,
            'data': sample_data.tolist(),
            'data_info': {
                'n_rois': sample_data.shape[1],
                'n_timepoints': sample_data.shape[0]
            }
        })
    
    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error in /api/sample-data: {error_msg}\n{traceback_str}")
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': traceback_str if app.debug else None
        }), 500


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='TwoTST API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    
    app.run(host=args.host, port=args.port, debug=args.debug)