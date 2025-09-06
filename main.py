import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
import traceback
from pathlib import Path
import sys

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# 导入自定义模块
try:
    from src.config.config import config
    from src.utils.data_processor import DrugAgeDataProcessor
    from src.utils.gpt_coordinator import GPTCoordinator
    from src.utils.query_analyzer import QueryAnalyzer
except ImportError as e:
    st.error(f"模块导入失败: {e}")
    st.error("请确保所有必要的文件都在正确的位置")
    st.error("当前项目结构应该是：")
    st.code("""
DRUGAGE/
├── src/
│   ├── config/
│   │   └── config.py
│   └── utils/
│       ├── data_processor.py
│       ├── gpt_coordinator.py
│       └── query_analyzer.py
├── data/
│   └── drugage.csv
└── main.py
    """)
    st.stop()

# 设置日志
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(**config.get_streamlit_config())

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background: linear-gradient(135deg, #f0f2f6 0%, #e8ecf3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton > button {
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
def initialize_session_state():
    """初始化session state变量"""
    defaults = {
        'data_processor': None,
        'gpt_coordinator': None,
        'query_analyzer': None,
        'chat_history': [],
        'system_initialized': False,
        'last_query_result': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# 主标题
st.markdown('<h1 class="main-header">🧬 DrugAge智能研究助手</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">专为延寿研究设计的AI驱动查询系统</p>', unsafe_allow_html=True)

# 侧边栏配置
with st.sidebar:
    st.title("🔧 系统配置")
    
    # API配置
    st.subheader("🤖 AI配置")
    api_key = st.text_input(
        "OpenAI API密钥",
        type="password",
        value=config.OPENAI_API_KEY or "",
        help="从 https://platform.openai.com/api-keys 获取"
    )
    
    model_choice = st.selectbox(
        "AI模型选择",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0,
        help="选择用于查询理解和响应生成的模型"
    )
    
    # 数据配置
    st.subheader("📊 数据配置")
    
    # 检查默认数据文件
    default_data_exists = config.DATA_PATH.exists()
    
    if default_data_exists:
        st.success(f"✅ 找到数据文件: {config.DATA_PATH.name}")
        use_default_data = st.checkbox("使用默认数据文件", value=True)
    else:
        st.warning("⚠️ 未找到默认数据文件")
        use_default_data = False
    
    uploaded_file = None
    if not use_default_data:
        uploaded_file = st.file_uploader(
            "上传DrugAge数据文件",
            type=['csv'],
            help="从 https://genomics.senescence.info/drugs/dataset.zip 下载"
        )
    
    # 系统初始化
    st.subheader("🚀 系统初始化")
    
    if st.button("🔄 初始化系统", type="primary"):
        with st.spinner("正在初始化系统..."):
            try:
                # 确定数据文件路径
                data_path = None
                if use_default_data and default_data_exists:
                    data_path = config.DATA_PATH
                elif uploaded_file is not None:
                    # 保存上传的文件
                    temp_data_path = config.DATA_DIR / "temp" / uploaded_file.name
                    temp_data_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(temp_data_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    data_path = temp_data_path
                
                if data_path is None:
                    st.error("请选择数据文件")
                    st.stop()
                
                # 初始化数据处理器
                st.session_state.data_processor = DrugAgeDataProcessor(str(data_path))
                
                # 初始化查询分析器
                st.session_state.query_analyzer = QueryAnalyzer()
                
                # 初始化GPT协调器
                if api_key:
                    st.session_state.gpt_coordinator = GPTCoordinator(api_key)
                    st.session_state.gpt_coordinator.model = model_choice
                    
                    # 注册工具函数
                    processor = st.session_state.data_processor
                    coordinator = st.session_state.gpt_coordinator
                    
                    coordinator.register_tool(
                        'search_drug', 
                        processor.search_drugs,
                        '搜索特定药物的信息'
                    )
                    coordinator.register_tool(
                        'compare_drugs',
                        processor.compare_drugs,
                        '比较多个药物的效果'
                    )
                    coordinator.register_tool(
                        'get_top_drugs',
                        processor.get_top_drugs,
                        '获取效果最好的药物'
                    )
                    coordinator.register_tool(
                        'analyze_effects',
                        lambda **kwargs: processor.get_top_drugs(50, **kwargs),
                        '分析药物效果'
                    )
                
                st.session_state.system_initialized = True
                st.success("✅ 系统初始化成功！")
                
            except Exception as e:
                st.error(f"❌ 初始化失败: {e}")
                logger.error(f"系统初始化失败: {e}")
                st.session_state.system_initialized = False
    
    # 系统状态显示
    st.subheader("📊 系统状态")
    
    status_components = {
        "数据处理器": st.session_state.data_processor is not None,
        "查询分析器": st.session_state.query_analyzer is not None,
        "GPT协调器": st.session_state.gpt_coordinator is not None,
        "系统就绪": st.session_state.system_initialized
    }
    
    for component, status in status_components.items():
        status_icon = "✅" if status else "❌"
        st.write(f"{component}: {status_icon}")
    
    # 数据统计
    if st.session_state.data_processor:
        st.subheader("📈 数据概况")
        try:
            stats = st.session_state.data_processor.generate_summary_stats()
            st.metric("总记录数", stats['total_records'])
            st.metric("唯一化合物", stats.get('unique_compounds', 'N/A'))
            st.metric("生物模型", stats.get('unique_organisms', 'N/A'))
        except Exception as e:
            st.warning(f"无法获取统计信息: {e}")

# 主界面内容
if not st.session_state.system_initialized:
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ 系统未初始化</h3>
        <p>请完成以下步骤来开始使用DrugAge智能助手：</p>
        <ol>
            <li>在侧边栏输入OpenAI API密钥</li>
            <li>选择数据文件（默认或上传）</li>
            <li>点击"初始化系统"按钮</li>
        </ol>
        <p><strong>数据获取：</strong></p>
        <p>1. 访问 <a href="https://genomics.senescence.info/drugs/" target="_blank">DrugAge数据库</a></p>
        <p>2. 下载 <a href="https://genomics.senescence.info/drugs/dataset.zip" target="_blank">数据集ZIP文件</a></p>
        <p>3. 解压并上传CSV文件</p>
    </div>
    """, unsafe_allow_html=True)
    
else:
    # 系统已初始化，显示主界面
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 智能对话", "🔍 药物搜索", "📊 数据分析", "🧠 查询分析", "📚 使用指南"
    ])
    
    with tab1:
        st.markdown('<h2 class="sub-header">💬 智能对话助手</h2>', unsafe_allow_html=True)
        
        # 查询建议按钮
        st.subheader("💡 快速查询")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 药物信息查询"):
                st.session_state.suggested_query = "告诉我关于rapamycin的详细信息"
        
        with col2:
            if st.button("⚖️ 药物比较"):
                st.session_state.suggested_query = "比较rapamycin和metformin的延寿效果"
        
        with col3:
            if st.button("📊 效果排名"):
                st.session_state.suggested_query = "显示延寿效果最好的前10种化合物"
        
        # 查询输入
        query_input = st.text_area(
            "请输入您的问题：",
            value=st.session_state.get('suggested_query', ''),
            placeholder="例如：哪些药物在小鼠中显示出显著的寿命延长效果？",
            height=100
        )
        
        # 清除建议查询
        if 'suggested_query' in st.session_state:
            del st.session_state.suggested_query
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🚀 提交查询", type="primary"):
                if query_input.strip():
                    with st.spinner("🤖 AI正在分析您的查询..."):
                        try:
                            # 使用GPT协调器处理查询
                            result = st.session_state.gpt_coordinator.process_query(query_input)
                            
                            # 保存结果
                            st.session_state.last_query_result = result
                            st.session_state.chat_history.append(result)
                            
                            # 显示结果
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.markdown("### 🤖 AI响应")
                            st.write(result['response'])
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # 显示分析详情
                            with st.expander("🔍 查询分析详情", expanded=False):
                                analysis = result['analysis']
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**查询意图:** {analysis['intent']}")
                                    st.write(f"**置信度:** {analysis['confidence']:.2f}")
                                
                                with col2:
                                    st.write(f"**识别实体:** {analysis['entities']}")
                                    st.write(f"**调用工具:** {analysis['tools_needed']}")
                                
                                # 工具执行结果
                                if result['tool_results']:
                                    st.write("**工具执行结果:**")
                                    for i, tool_result in enumerate(result['tool_results']):
                                        status = "✅" if tool_result.success else "❌"
                                        st.write(f"{i+1}. {status} {'成功' if tool_result.success else tool_result.error_message}")
                        
                        except Exception as e:
                            st.error(f"查询处理失败: {e}")
                            logger.error(f"查询处理错误: {e}")
                            traceback.print_exc()
                else:
                    st.warning("请输入查询内容")
        
        with col2:
            if st.button("🗑️ 清除历史"):
                st.session_state.chat_history = []
                if st.session_state.gpt_coordinator:
                    st.session_state.gpt_coordinator.clear_history()
                st.success("历史记录已清除")
                st.rerun()
        
        # 显示对话历史
        if st.session_state.chat_history:
            st.subheader("💭 对话历史")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"对话 {len(st.session_state.chat_history)-i}: {chat['query'][:50]}..."):
                    st.markdown(f"**用户:** {chat['query']}")
                    st.markdown(f"**助手:** {chat['response']}")
                    if chat.get('timestamp'):
                        st.write(f"**时间:** {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    with tab2:
        st.markdown('<h2 class="sub-header">🔍 药物搜索</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("搜索参数")
            drug_name = st.text_input("药物名称", placeholder="例如：rapamycin")
            exact_match = st.checkbox("精确匹配", value=False)
            min_effect = st.number_input("最小效果值(%)", min_value=0.0, value=0.0, step=1.0)
            
            if st.button("🔍 搜索", type="primary"):
                if drug_name.strip():
                    with st.spinner("搜索中..."):
                        try:
                            results = st.session_state.data_processor.search_drugs(
                                drug_name, exact_match, min_effect
                            )
                            
                            with col2:
                                st.subheader("搜索结果")
                                
                                if len(results) > 0:
                                    st.success(f"找到 {len(results)} 条记录")
                                    
                                    # 显示结果表格
                                    st.dataframe(results.head(10), use_container_width=True)
                                    
                                    # 基本统计
                                    if len(results) > 1:
                                        st.subheader("统计信息")
                                        
                                        column_map = st.session_state.data_processor.column_map
                                        if 'lifespan_effect' in column_map:
                                            effect_col = column_map['lifespan_effect']
                                            effects = pd.to_numeric(results[effect_col], errors='coerce').dropna()
                                            
                                            if len(effects) > 0:
                                                col1_stats, col2_stats, col3_stats = st.columns(3)
                                                
                                                with col1_stats:
                                                    st.metric("平均效果", f"{effects.mean():.1f}%")
                                                
                                                with col2_stats:
                                                    st.metric("最大效果", f"{effects.max():.1f}%")
                                                
                                                with col3_stats:
                                                    st.metric("研究数量", len(effects))
                                                
                                                # 效果分布图
                                                if len(effects) > 1:
                                                    fig = px.histogram(
                                                        x=effects,
                                                        title=f"{drug_name} 效果分布",
                                                        labels={'x': '寿命延长(%)', 'y': '研究数量'},
                                                        nbins=min(20, len(effects))
                                                    )
                                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning(f"未找到关于 '{drug_name}' 的记录")
                                    
                                    # 提供建议
                                    if st.session_state.gpt_coordinator:
                                        suggestions = st.session_state.gpt_coordinator.get_suggestions(drug_name)
                                        if suggestions:
                                            st.subheader("建议查询:")
                                            for suggestion in suggestions[:3]:
                                                st.write(f"• {suggestion}")
                        
                        except Exception as e:
                            st.error(f"搜索失败: {e}")
                            logger.error(f"搜索错误: {e}")
        
        # 批量搜索
        st.subheader("📦 批量搜索")
        batch_input = st.text_area(
            "输入多个药物名称（每行一个）：",
            placeholder="rapamycin\nmetformin\nresveratrol",
            height=100
        )
        
        if st.button("🚀 批量搜索"):
            if batch_input.strip():
                drug_list = [drug.strip() for drug in batch_input.split('\n') if drug.strip()]
                
                if drug_list:
                    with st.spinner(f"正在搜索 {len(drug_list)} 个药物..."):
                        try:
                            batch_results = {}
                            for drug in drug_list:
                                results = st.session_state.data_processor.search_drugs(drug)
                                batch_results[drug] = results
                            
                            # 显示批量结果
                            success_count = sum(1 for results in batch_results.values() if len(results) > 0)
                            st.success(f"成功找到 {success_count}/{len(drug_list)} 个药物的数据")
                            
                            # 创建汇总表
                            summary_data = []
                            column_map = st.session_state.data_processor.column_map
                            effect_col = column_map.get('lifespan_effect', '')
                            
                            for drug_name, results in batch_results.items():
                                if len(results) > 0:
                                    if effect_col and effect_col in results.columns:
                                        effects = pd.to_numeric(results[effect_col], errors='coerce').dropna()
                                        if len(effects) > 0:
                                            summary_data.append({
                                                '药物': drug_name,
                                                '研究数量': len(results),
                                                '平均效果(%)': f"{effects.mean():.1f}",
                                                '最大效果(%)': f"{effects.max():.1f}",
                                                '最小效果(%)': f"{effects.min():.1f}"
                                            })
                                        else:
                                            summary_data.append({
                                                '药物': drug_name,
                                                '研究数量': len(results),
                                                '平均效果(%)': 'N/A',
                                                '最大效果(%)': 'N/A',
                                                '最小效果(%)': 'N/A'
                                            })
                                else:
                                    summary_data.append({
                                        '药物': drug_name,
                                        '研究数量': 0,
                                        '平均效果(%)': 'N/A',
                                        '最大效果(%)': 'N/A',
                                        '最小效果(%)': 'N/A'
                                    })
                            
                            if summary_data:
                                st.subheader("搜索汇总")
                                summary_df = pd.DataFrame(summary_data)
                                st.dataframe(summary_df, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"批量搜索失败: {e}")
                            logger.error(f"批量搜索错误: {e}")
    
    with tab3:
        st.markdown('<h2 class="sub-header">📊 数据分析</h2>', unsafe_allow_html=True)
        
        try:
            # 获取统计信息
            stats = st.session_state.data_processor.generate_summary_stats()
            
            # 关键指标展示
            st.subheader("📈 关键指标")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 {stats['total_records']}</h3>
                    <p>总研究记录</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💊 {stats.get('unique_compounds', 'N/A')}</h3>
                    <p>唯一化合物</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🐭 {stats.get('unique_organisms', 'N/A')}</h3>
                    <p>生物模型</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                effect_stats = stats.get('effect_statistics', {})
                avg_effect = effect_stats.get('mean', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 {avg_effect:.1f}%</h3>
                    <p>平均延寿效果</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 详细分析
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔬 研究最多的化合物")
                if stats.get('top_studied_compounds'):
                    top_compounds = stats['top_studied_compounds']
                    compounds_df = pd.DataFrame([
                        {'化合物': k, '研究数量': v} 
                        for k, v in list(top_compounds.items())[:10]
                    ])
                    
                    fig_compounds = px.bar(
                        compounds_df, 
                        x='研究数量', 
                        y='化合物',
                        orientation='h',
                        title="研究数量Top 10化合物"
                    )
                    st.plotly_chart(fig_compounds, use_container_width=True)
            
            with col2:
                st.subheader("🧬 生物模型分布")
                if stats.get('organism_distribution'):
                    org_dist = stats['organism_distribution']
                    org_df = pd.DataFrame([
                        {'生物模型': k, '研究数量': v} 
                        for k, v in list(org_dist.items())[:8]
                    ])
                    
                    fig_organisms = px.pie(
                        org_df,
                        values='研究数量',
                        names='生物模型',
                        title="生物模型研究分布"
                    )
                    st.plotly_chart(fig_organisms, use_container_width=True)
            
            # 效果分析
            if effect_stats:
                st.subheader("📈 延寿效果分析")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("最大效果", f"{effect_stats.get('max', 0):.1f}%")
                    st.metric("最小效果", f"{effect_stats.get('min', 0):.1f}%")
                
                with col2:
                    st.metric("中位数效果", f"{effect_stats.get('median', 0):.1f}%")
                    st.metric("标准差", f"{effect_stats.get('std', 0):.1f}%")
                
                with col3:
                    st.metric("正效果研究", effect_stats.get('positive_effects', 0))
                    st.metric("负效果研究", effect_stats.get('negative_effects', 0))
                
                # 效果分布直方图
                top_drugs = st.session_state.data_processor.get_top_drugs(100)
                if len(top_drugs) > 0:
                    column_map = st.session_state.data_processor.column_map
                    effect_col = column_map.get('lifespan_effect', '')
                    
                    if effect_col in top_drugs.columns:
                        effects = pd.to_numeric(top_drugs[effect_col], errors='coerce').dropna()
                        
                        fig_hist = px.histogram(
                            x=effects,
                            nbins=30,
                            title="延寿效果分布直方图",
                            labels={'x': '寿命延长(%)', 'y': '研究数量'}
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
        
        except Exception as e:
            st.error(f"数据分析失败: {e}")
            logger.error(f"数据分析错误: {e}")
    
    with tab4:
        st.markdown('<h2 class="sub-header">🧠 查询分析器</h2>', unsafe_allow_html=True)
        
        # 查询分析工具
        st.subheader("🔍 智能查询分析")
        
        test_query = st.text_input(
            "输入查询进行分析：",
            placeholder="例如：比较rapamycin和metformin在小鼠中的效果"
        )
        
        if st.button("🔬 分析查询"):
            if test_query.strip():
                try:
                    context = st.session_state.query_analyzer.analyze_query(test_query)
                    
                    # 显示分析结果
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("查询理解")
                        st.write(f"**原始查询:** {test_query}")
                        st.write(f"**检测类型:** {context.query_type.value}")
                        st.write(f"**主要意图:** {context.primary_intent}")
                        st.write(f"**置信度:** {context.confidence:.2f}")
                    
                    with col2:
                        st.subheader("提取信息")
                        st.json(context.entities)
                        
                        if context.parameters:
                            st.subheader("参数")
                            st.json(context.parameters)
                    
                    # 建议
                    if context.suggestions:
                        st.subheader("💡 改进建议")
                        for suggestion in context.suggestions:
                            st.write(f"• {suggestion}")
                
                except Exception as e:
                    st.error(f"查询分析失败: {e}")
        
        # 查询示例展示
        st.subheader("💡 查询示例")
        
        if st.session_state.query_analyzer:
            try:
                examples = st.session_state.query_analyzer.get_query_examples()
                
                for category, example_list in examples.items():
                    with st.expander(f"📝 {category}"):
                        for example in example_list:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"• {example}")
                            with col2:
                                if st.button(f"分析", key=f"analyze_{example[:20]}"):
                                    context = st.session_state.query_analyzer.analyze_query(example)
                                    st.json({
                                        'type': context.query_type.value,
                                        'confidence': context.confidence,
                                        'entities': context.entities
                                    })
            
            except Exception as e:
                st.warning(f"无法加载查询示例: {e}")
    
    with tab5:
        st.markdown('<h2 class="sub-header">📚 使用指南</h2>', unsafe_allow_html=True)
        
        # 使用指南内容
        st.markdown("""
        <div class="info-box">
            <h3>🚀 快速开始</h3>
            <p>DrugAge智能助手是专为延寿研究设计的AI驱动查询系统，帮助研究人员快速查询和比较药物的寿命延长效果。</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 功能介绍
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 主要功能")
            
            st.write("""
            **💬 智能对话**
            - 自然语言查询理解
            - 智能工具调用
            - 上下文感知响应
            
            **🔍 药物搜索**
            - 精确和模糊匹配
            - 批量搜索支持
            - 详细药物信息
            
            **📊 数据分析**
            - 数据集概览统计
            - 研究趋势分析
            - 可视化图表展示
            
            **🧠 查询分析**
            - 自然语言理解
            - 实体识别展示
            - 查询优化建议
            """)
        
        with col2:
            st.subheader("💡 使用技巧")
            
            st.write("""
            **查询优化建议：**
            - 使用具体的药物名称
            - 指定感兴趣的生物模型
            - 明确查询意图（搜索、比较、排名）
            
            **高效使用方法：**
            - 先搜索单个药物了解基础信息
            - 使用智能对话进行复杂查询
            - 结合数据分析了解研究现状
            
            **注意事项：**
            - 所有结果基于实验室研究数据
            - 不提供医疗建议或临床推荐
            - 建议查阅原始文献获取完整信息
            """)
        
        # 查询示例
        st.subheader("📝 查询示例")
        
        examples_data = {
            "药物信息查询": [
                "告诉我关于rapamycin的详细信息",
                "metformin的延寿机制是什么？",
                "resveratrol在哪些生物模型中进行了测试？"
            ],
            "效果分析": [
                "哪些药物的延寿效果超过20%？",
                "显示延寿效果最好的前10种化合物",
                "在小鼠中测试的药物中哪个效果最好？"
            ],
            "比较分析": [
                "比较rapamycin和metformin的延寿效果",
                "rapamycin vs resveratrol vs curcumin",
                "哪个更有效：aspirin还是lithium？"
            ],
            "特定条件查询": [
                "在C. elegans中测试过的所有药物",
                "延长寿命超过30%的化合物有哪些？",
                "研究数量最多的前5种药物"
            ]
        }
        
        for category, examples in examples_data.items():
            with st.expander(f"📂 {category}"):
                for example in examples:
                    st.write(f"💬 {example}")
        
        # 数据来源信息
        st.subheader("📖 数据来源")
        st.markdown("""
        <div class="info-box">
            <h4>DrugAge数据库</h4>
            <p><strong>来源：</strong> <a href="https://genomics.senescence.info/drugs/" target="_blank">Human Ageing Genomic Resources (HAGR)</a></p>
            <p><strong>描述：</strong> DrugAge是一个专门收集与延长寿命相关的化合物实验数据的数据库</p>
            <p><strong>数据特点：</strong></p>
            <ul>
                <li>涵盖多种模式生物（小鼠、大鼠、线虫、果蝇、酵母等）</li>
                <li>包含化合物名称、剂量、效果、实验条件等详细信息</li>
                <li>定期更新，反映最新的延寿研究进展</li>
                <li>所有数据都有文献引用，确保可追溯性</li>
            </ul>
            <p><strong>引用：</strong> Barardo, D. et al. The DrugAge database of aging-related drugs. Aging Cell 16, 594–597 (2017).</p>
        </div>
        """, unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown("""
<div class="info-box">
    <h4>📞 关于DrugAge智能助手</h4>
    <p>本系统专为延寿研究人员和生命科学专业人士设计，基于DrugAge数据库提供智能化的药物查询和分析服务。</p>
    <p><strong>⚠️ 重要声明：</strong> 本工具仅用于科研目的，所有信息均基于实验室研究，不构成医疗建议。使用前请咨询专业医疗人员。</p>
    <p><strong>🔗 相关链接：</strong></p>
    <p>• <a href="https://genomics.senescence.info/drugs/" target="_blank">DrugAge数据库官网</a></p>
    <p>• <a href="https://platform.openai.com/" target="_blank">OpenAI平台</a></p>
    <p>• <a href="https://streamlit.io/" target="_blank">Streamlit框架</a></p>
</div>
""", unsafe_allow_html=True)

# 运行时检查
if __name__ == "__main__":
    st.write("🚀 DrugAge智能助手正在运行...")
    
    # 配置验证
    config_errors = config.validate_config()
    if config_errors:
        st.sidebar.error("配置验证失败:")
        for error in config_errors:
            st.sidebar.error(f"• {error}")
    else:
        st.sidebar.success("✅ 配置验证通过")