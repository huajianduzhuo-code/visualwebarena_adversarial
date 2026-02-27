#!/bin/bash
###############################################################################
# setup_vwa.sh  –  VisualWebArena + Adversarial Attack 一键配置脚本
#
# 在新机器上 clone 完 repo 后运行此脚本即可自动配置好评估环境。
#
# 前置条件:
#   - conda 已安装 (miniconda / anaconda)
#   - Docker 已安装且当前用户有 Docker 权限 (或能 sudo)
#   - GPU 机器 (至少 1 块 GPU; 多 GPU 可加速推理)
#   - 至少 50 GB 可用磁盘 (Docker 镜像 ~12GB + 模型)
#
# 使用方法:
#   chmod +x setup_vwa.sh
#
#   # 一键完成所有配置 (Step 1 – 4):
#   ./setup_vwa.sh all
#
#   # 或者分步执行:
#   ./setup_vwa.sh step1   # Python 环境安装 + 依赖升级
#   ./setup_vwa.sh step2   # 代码 patch (支持自定义多模态模型)
#   ./setup_vwa.sh step3   # Docker 网站环境搭建
#   ./setup_vwa.sh step4   # 环境变量 + 生成任务配置 + 登录 Cookie
#   ./setup_vwa.sh step5   # (可选) 在 tmux 中启动 vLLM 模型服务
#
# 配置完成后运行评估:
#   # adversarial evaluation (100 tasks, parallel):
#   bash run_adversarial_eval.sh --agent /path/to/agent_model --attacker /path/to/attacker_model
#
#   # 或 clean evaluation (无攻击):
#   bash run_adversarial_eval.sh --agent /path/to/model --no-attacker
#
#   # 详见: bash run_adversarial_eval.sh --help
###############################################################################

set -e  # 遇错即停

###############################################################################
# 用户配置 (请根据实际情况修改)
###############################################################################

# ---- 网站服务器配置 ----
# Docker 网站运行的服务器地址（可以是当前机器的 IP 或 localhost）
# 注意: 如果用 localhost，Docker 容器内的 URL 也会用 localhost，
#        跨机器评估时请改为实际 IP。
SERVER_HOSTNAME="${SERVER_HOSTNAME:-localhost}"

# ---- vLLM 配置 (Step 5 使用) ----
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_API_KEY="${VLLM_API_KEY:-dummy}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-4}"              # tensor parallel size
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-spag}"       # vLLM 运行在哪个 conda 环境

# ---- 模型配置 (Step 5 使用) ----
MODEL_PATH="${MODEL_PATH:-}"
MODEL_NAME="${MODEL_NAME:-model}"

# ---- VWA 项目路径 ----
VWA_DIR="${VWA_DIR:-$(cd "$(dirname "$0")" && pwd)}"

# ---- Conda 环境名 ----
CONDA_ENV_NAME="vwa"

###############################################################################
# 颜色输出
###############################################################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "\n${BLUE}========== $1 ==========${NC}\n"; }

###############################################################################
# Helper: 确保 conda 命令可用
###############################################################################
ensure_conda() {
    if ! command -v conda &>/dev/null; then
        # Try sourcing conda.sh from common locations
        for CONDA_SH in \
            "$HOME/miniconda3/etc/profile.d/conda.sh" \
            "$HOME/anaconda3/etc/profile.d/conda.sh" \
            "/opt/conda/etc/profile.d/conda.sh"; do
            if [ -f "$CONDA_SH" ]; then
                source "$CONDA_SH"
                break
            fi
        done
    fi
    if ! command -v conda &>/dev/null; then
        log_error "conda 未找到。请先安装 miniconda/anaconda。"
        exit 1
    fi
}

###############################################################################
# Step 1: Python 环境安装
###############################################################################
step1_python_env() {
    log_step "Step 1: Python 环境安装"
    cd "$VWA_DIR"
    ensure_conda

    # 创建独立的 conda 环境 (不污染其他环境)
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        log_info "conda 环境 '${CONDA_ENV_NAME}' 已存在，激活中..."
    else
        log_info "创建 conda 环境 '${CONDA_ENV_NAME}' (Python 3.11)..."
        conda create -n "${CONDA_ENV_NAME}" python=3.11 -y 2>&1 | tail -3
    fi
    conda activate "${CONDA_ENV_NAME}"
    log_info "当前 Python: $(which python3) ($(python3 --version))"

    # 1) 安装 VWA 原始依赖
    log_info "安装 requirements.txt 依赖..."
    pip install -r requirements.txt 2>&1 | tail -5

    # 2) 升级关键包到兼容版本
    #    VWA 的 requirements.txt 锁定了一些老版本，与 BLIP2、vLLM 等不兼容。
    #    以下升级已在实践中验证可用:
    log_info "升级关键 Python 包..."

    # PyTorch: VWA 锁定 2.0.1 太老，transformers>=4.40 需要 torch>=2.1
    pip install torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -3

    # transformers + tokenizers: BLIP2 需要较新版本
    pip install transformers==4.57.6 tokenizers==0.22.2 2>&1 | tail -3

    # huggingface-hub: 需与 transformers 匹配
    pip install huggingface-hub==0.36.2 2>&1 | tail -3

    # 3) 安装 Playwright 浏览器
    log_info "安装 Playwright 浏览器..."
    playwright install chromium 2>&1 | tail -3

    # 4) 安装 VWA 包本身
    log_info "安装 visualwebarena 包 (editable mode)..."
    pip install -e . 2>&1 | tail -3

    # 5) 把当前用户加入 docker 组 (如果还没有)
    if ! id -nG | grep -qw docker; then
        log_info "将当前用户加入 docker 组..."
        sudo usermod -aG docker "$(whoami)" 2>/dev/null || log_warn "无法加入 docker 组，请手动执行: sudo usermod -aG docker \$(whoami)"
        log_warn "加入 docker 组后需要重新登录或运行: newgrp docker"
    fi

    # 6) 验证安装
    log_info "验证安装..."
    DATASET=visualwebarena \
    CLASSIFIEDS="http://placeholder:9980" CLASSIFIEDS_RESET_TOKEN="placeholder" \
    SHOPPING="http://placeholder:7770" REDDIT="http://placeholder:9999" \
    WIKIPEDIA="http://placeholder:8888" HOMEPAGE="http://placeholder:4399" \
    OPENAI_API_KEY="placeholder" \
    python3 -c "
from browser_env import ScriptBrowserEnv; print('✓ browser_env 导入成功')
from agent import construct_agent; print('✓ agent 导入成功')
import openai; print(f'✓ openai {openai.__version__}')
import tiktoken; print('✓ tiktoken OK')
import torch; print(f'✓ torch {torch.__version__}')
import transformers; print(f'✓ transformers {transformers.__version__}')
print('所有依赖安装验证通过!')
"

    log_info "Step 1 完成 ✓"
    log_info "后续使用前请先: conda activate ${CONDA_ENV_NAME}"
}

###############################################################################
# Step 2: 代码 Patch - 支持自定义多模态模型 + 对抗评估
###############################################################################
step2_code_patch() {
    log_step "Step 2: 代码 Patch"
    cd "$VWA_DIR"

    # -------------------------------------------------------------------------
    # 2a: agent/agent.py  – 通用多模态支持
    # -------------------------------------------------------------------------
    log_info "[2a] 修改 agent/agent.py: 启用通用多模态支持..."

    if grep -q "# PATCHED: universal multimodal support" agent/agent.py 2>/dev/null; then
        log_warn "  agent/agent.py 已修改，跳过"
    else
        python3 << 'PATCH_AGENT'
with open("agent/agent.py", "r") as f:
    content = f.read()

old = '''        # Check if the model is multimodal.
        if ("gemini" in lm_config.model or "gpt-4" in lm_config.model and "vision" in lm_config.model) and type(prompt_constructor) == MultimodalCoTPromptConstructor:
            self.multimodal_inputs = True
        else:
            self.multimodal_inputs = False'''

new = '''        # PATCHED: universal multimodal support
        # Enable multimodal inputs for ANY model when using MultimodalCoTPromptConstructor
        # This allows local models (gemma-3, qwen-vl, llama-vision, etc.) to work in multimodal mode
        if type(prompt_constructor) == MultimodalCoTPromptConstructor:
            self.multimodal_inputs = True
        else:
            self.multimodal_inputs = False'''

if old in content:
    content = content.replace(old, new)
    with open("agent/agent.py", "w") as f:
        f.write(content)
    print("  ✓ agent/agent.py patched")
else:
    print("  ⚠ agent/agent.py: pattern not found (may already be patched)")
PATCH_AGENT
    fi

    # -------------------------------------------------------------------------
    # 2b: llms/tokenizers.py  – tokenizer fallback
    # -------------------------------------------------------------------------
    log_info "[2b] 修改 llms/tokenizers.py: 添加 tokenizer fallback..."

    if grep -q "# PATCHED: fallback tokenizer" llms/tokenizers.py 2>/dev/null; then
        log_warn "  llms/tokenizers.py 已修改，跳过"
    else
        python3 << 'PATCH_TOKENIZER'
with open("llms/tokenizers.py", "r") as f:
    content = f.read()

old = '''    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
            self.tokenizer = tiktoken.encoding_for_model(model_name)'''

new = '''    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
            # PATCHED: fallback tokenizer for non-OpenAI models served via OpenAI-compatible API
            try:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Use cl100k_base as fallback (GPT-4 tokenizer, reasonable approximation)
                self.tokenizer = tiktoken.get_encoding("cl100k_base")'''

if old in content:
    content = content.replace(old, new)
    with open("llms/tokenizers.py", "w") as f:
        f.write(content)
    print("  ✓ llms/tokenizers.py patched")
else:
    print("  ⚠ llms/tokenizers.py: pattern not found (may already be patched)")
PATCH_TOKENIZER
    fi

    # -------------------------------------------------------------------------
    # 2c: agent/prompts/prompt_constructor.py  – chat template 角色交替
    #     Gemma/Llama 模型要求严格的 user/assistant 交替，不支持 system 角色。
    #     原代码使用连续的 system role 会导致 vLLM TemplateError。
    # -------------------------------------------------------------------------
    log_info "[2c] 修改 prompt_constructor.py: 支持 Gemma/Llama 角色交替..."

    if grep -q "is_strict_alternation" agent/prompts/prompt_constructor.py 2>/dev/null; then
        log_warn "  prompt_constructor.py 已修改，跳过"
    else
        python3 << 'PATCH_PROMPT'
with open("agent/prompts/prompt_constructor.py", "r") as f:
    content = f.read()

# ---- Patch PromptConstructor.get_lm_api_input ----
old_text = '''            if self.lm_config.mode == "chat":
                message = [{"role": "system", "content": intro}]
                for (x, y) in examples:
                    message.append(
                        {
                            "role": "system",
                            "name": "example_user",
                            "content": x,
                        }
                    )
                    message.append(
                        {
                            "role": "system",
                            "name": "example_assistant",
                            "content": y,
                        }
                    )
                message.append({"role": "user", "content": current})'''

new_text = '''            if self.lm_config.mode == "chat":
                # Check if model requires strict user/assistant alternation
                model_lower = self.lm_config.model.lower()
                is_strict_alternation = "gemma" in model_lower or "llama" in model_lower

                if is_strict_alternation:
                    # For Gemma/Llama: use user/assistant alternation,
                    # prepend intro to the first user message
                    message = []
                    for i, (x, y) in enumerate(examples):
                        user_text = (intro + "\\n\\n" + x) if i == 0 else x
                        message.append({"role": "user", "content": user_text})
                        message.append({"role": "assistant", "content": y})
                    # If no examples, prepend intro to current
                    if not examples:
                        message.append({"role": "user", "content": intro + "\\n\\n" + current})
                    else:
                        message.append({"role": "user", "content": current})
                else:
                    # Original behavior for OpenAI models
                    message = [{"role": "system", "content": intro}]
                    for (x, y) in examples:
                        message.append(
                            {
                                "role": "system",
                                "name": "example_user",
                                "content": x,
                            }
                        )
                        message.append(
                            {
                                "role": "system",
                                "name": "example_assistant",
                                "content": y,
                            }
                        )
                    message.append({"role": "user", "content": current})'''

if old_text in content:
    content = content.replace(old_text, new_text)
    print("  ✓ PromptConstructor.get_lm_api_input patched")
else:
    print("  ⚠ PromptConstructor text-only pattern not found")

# ---- Patch MultimodalCoTPromptConstructor.get_lm_api_input ----
old_mm = '''            if self.lm_config.mode == "chat":
                message = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": intro}],
                    }
                ]
                for (x, y, z) in examples:
                    example_img = Image.open(z)
                    message.append(
                        {
                            "role": "system",
                            "name": "example_user",
                            "content": [
                                {"type": "text", "text": x},
                                {
                                    "type": "text",
                                    "text": "IMAGES: (1) current page screenshot",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": pil_to_b64(example_img)
                                    },
                                },
                            ],
                        }
                    )
                    message.append(
                        {
                            "role": "system",
                            "name": "example_assistant",
                            "content": [{"type": "text", "text": y}],
                        }
                    )'''

new_mm = '''            if self.lm_config.mode == "chat":
                # Check if model requires strict user/assistant alternation
                # (e.g., Gemma models don't support system role or named roles)
                model_lower = self.lm_config.model.lower()
                is_strict_alternation = "gemma" in model_lower or "llama" in model_lower

                if is_strict_alternation:
                    # For Gemma/Llama models: strict user/assistant alternation
                    # Prepend intro to the first user message
                    message = []
                    for i, (x, y, z) in enumerate(examples):
                        example_img = Image.open(z)
                        user_text = (intro + "\\n\\n" + x) if i == 0 else x
                        user_content = [
                            {"type": "text", "text": user_text},
                            {
                                "type": "text",
                                "text": "IMAGES: (1) current page screenshot",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": pil_to_b64(example_img)
                                },
                            },
                        ]
                        message.append({"role": "user", "content": user_content})
                        message.append(
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": y}],
                            }
                        )

                    # Build final user message with current observation + images
                    current_prompt = current
                    # If no examples were provided, prepend intro here
                    if not examples:
                        current_prompt = intro + "\\n\\n" + current_prompt

                    content = [
                        {
                            "type": "text",
                            "text": "IMAGES: (1) current page screenshot",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64(page_screenshot_img)},
                        },
                    ]
                    for image_i, image in enumerate(images):
                        content.extend(
                            [
                                {
                                    "type": "text",
                                    "text": f"({image_i+2}) input image {image_i+1}",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": pil_to_b64(image)},
                                },
                            ]
                        )
                    content = [{"type": "text", "text": current_prompt}] + content
                    message.append({"role": "user", "content": content})
                    return message

                else:
                    # Original behavior for GPT-4V and similar models
                    # that support system role with named messages
                    message = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": intro}],
                        }
                    ]
                    for (x, y, z) in examples:
                        example_img = Image.open(z)
                        message.append(
                            {
                                "role": "system",
                                "name": "example_user",
                                "content": [
                                    {"type": "text", "text": x},
                                    {
                                        "type": "text",
                                        "text": "IMAGES: (1) current page screenshot",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": pil_to_b64(example_img)
                                        },
                                    },
                                ],
                            }
                        )
                        message.append(
                            {
                                "role": "system",
                                "name": "example_assistant",
                                "content": [{"type": "text", "text": y}],
                            }
                        )'''

if old_mm in content:
    content = content.replace(old_mm, new_mm)
    print("  ✓ MultimodalCoTPromptConstructor.get_lm_api_input patched")
else:
    print("  ⚠ MultimodalCoTPromptConstructor multimodal pattern not found")

with open("agent/prompts/prompt_constructor.py", "w") as f:
    f.write(content)

PATCH_PROMPT
    fi

    # -------------------------------------------------------------------------
    # 2d: run.py  – 支持 --task_ids 参数
    # -------------------------------------------------------------------------
    log_info "[2d] 修改 run.py: 添加 --task_ids 参数支持..."

    if grep -q "task_ids" run.py 2>/dev/null; then
        log_warn "  run.py 已修改，跳过"
    else
        python3 << 'PATCH_RUN'
with open("run.py", "r") as f:
    content = f.read()

# Add --task_ids argument after --test_end_idx
old_arg = '    parser.add_argument("--test_end_idx", type=int, default=910)'
new_arg = '''    parser.add_argument("--test_end_idx", type=int, default=910)
    parser.add_argument(
        "--task_ids",
        type=str,
        default="",
        help="Comma-separated list of specific task IDs to run (overrides start/end idx)",
    )'''

if old_arg in content:
    content = content.replace(old_arg, new_arg)
    print("  ✓ run.py: --task_ids argument added")
else:
    print("  ⚠ run.py: argument insertion point not found")

# Patch the test_file_list construction to support task_ids
old_list = '''    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        test_file_list.append(os.path.join(test_config_base_dir, f"{i}.json"))'''

new_list = '''    test_file_list = []
    if args.task_ids:
        # Use explicit task IDs
        for tid in args.task_ids.split(","):
            tid = tid.strip()
            if tid:
                test_file_list.append(os.path.join(test_config_base_dir, f"{tid}.json"))
    else:
        st_idx = args.test_start_idx
        ed_idx = args.test_end_idx
        for i in range(st_idx, ed_idx):
            test_file_list.append(os.path.join(test_config_base_dir, f"{i}.json"))'''

if old_list in content:
    content = content.replace(old_list, new_list)
    print("  ✓ run.py: task_ids file list logic patched")
else:
    print("  ⚠ run.py: file list pattern not found")

with open("run.py", "w") as f:
    f.write(content)

PATCH_RUN
    fi

    # -------------------------------------------------------------------------
    # 2e: 生成 prompt JSON 文件 (如果 jsons 目录为空)
    # -------------------------------------------------------------------------
    log_info "[2e] 生成 prompt JSON 文件..."
    if [ -f "agent/prompts/jsons/p_som_cot_id_actree_3s.json" ]; then
        log_warn "  prompt JSON 已存在，跳过"
    else
        cd agent/prompts
        python3 to_json.py
        cd "$VWA_DIR"
        log_info "  prompt JSON 已生成"
    fi

    log_info "Step 2 完成 ✓"
}

###############################################################################
# Step 3: Docker 网站环境搭建
###############################################################################
step3_docker_setup() {
    log_step "Step 3: Docker 网站环境搭建"

    log_info "在服务器 ${SERVER_HOSTNAME} 上搭建 VWA 网站环境"
    log_info "注意: 需要较大磁盘空间(~50GB+)和 Docker 权限"

    # 检查 docker 是否可用
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        return 1
    fi

    # ---- 3a: 下载所有镜像 (使用 aria2c 多线程加速) ----
    log_info "=== 下载 Docker 镜像 ==="
    if ! command -v aria2c &> /dev/null; then
        log_info "安装 aria2c..."
        sudo apt-get install -y aria2 2>/dev/null || sudo yum install -y aria2 2>/dev/null || {
            log_warn "无法安装 aria2，将使用 wget 代替（较慢）"
        }
    fi

    DOWNLOAD_DIR="/tmp/vwa_images"
    mkdir -p "$DOWNLOAD_DIR"

    # 通用下载函数：优先 aria2c，fallback 到 wget
    download_file() {
        local output="$1"
        shift
        local urls=("$@")
        if [ -f "$DOWNLOAD_DIR/$output" ]; then
            log_info "  $output 已存在，跳过下载"
            return 0
        fi
        if command -v aria2c &> /dev/null; then
            aria2c -x16 -s16 -k1M --file-allocation=none -d "$DOWNLOAD_DIR" -o "$output" "${urls[@]}"
        else
            wget -q --show-progress -O "$DOWNLOAD_DIR/$output" "${urls[0]}"
        fi
    }

    # Reddit (~2-3GB)
    download_file "postmill.tar" \
        "http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar" \
        "https://archive.org/download/postmill-populated-exposed-withimg/postmill-populated-exposed-withimg.tar"

    # Shopping (~4-5GB)
    download_file "shopping_final_0712.tar" \
        "http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar" \
        "https://archive.org/download/webarena-env-shopping-image/shopping_final_0712.tar"

    # Classifieds (docker compose zip)
    download_file "classifieds_docker_compose.zip" \
        "https://archive.org/download/classifieds_docker_compose/classifieds_docker_compose.zip"

    # Wikipedia (可选，很大 ~90GB，通过 SKIP_WIKIPEDIA 控制)
    if [ "${SKIP_WIKIPEDIA:-1}" = "0" ]; then
        WIKI_DATA_DIR="$VWA_DIR/wiki_data"
        mkdir -p "$WIKI_DATA_DIR"
        download_file "wikipedia_en_all_maxi_2022-05.zim" \
            "http://metis.lti.cs.cmu.edu/webarena-images/wikipedia_en_all_maxi_2022-05.zim" \
            "https://archive.org/download/webarena-env-wiki-image/wikipedia_en_all_maxi_2022-05.zim"
        mv "$DOWNLOAD_DIR/wikipedia_en_all_maxi_2022-05.zim" "$WIKI_DATA_DIR/" 2>/dev/null || true
    else
        log_warn "跳过 Wikipedia 下载 (设置 SKIP_WIKIPEDIA=0 可启用，需要 ~90GB 磁盘)"
    fi

    # ---- 3b: Reddit 网站 (端口 9999) ----
    log_info "=== 设置 Reddit 网站 ==="
    if docker ps --format '{{.Names}}' | grep -q "forum"; then
        log_warn "Reddit (forum) 容器已运行，跳过"
    else
        if ! docker images --format '{{.Repository}}' | grep -q "postmill-populated-exposed-withimg"; then
            log_info "导入 Reddit 镜像..."
            docker load --input "$DOWNLOAD_DIR/postmill.tar"
        fi
        docker run --name forum -p 9999:80 -d postmill-populated-exposed-withimg
        log_info "Reddit 启动完成: http://${SERVER_HOSTNAME}:9999"
    fi

    # ---- 3c: Shopping 网站 (端口 7770) ----
    log_info "=== 设置 Shopping 网站 ==="
    if docker ps --format '{{.Names}}' | grep -q "shopping"; then
        log_warn "Shopping 容器已运行，跳过"
    else
        if ! docker images --format '{{.Repository}}' | grep -q "shopping_final_0712"; then
            log_info "导入 Shopping 镜像..."
            docker load --input "$DOWNLOAD_DIR/shopping_final_0712.tar"
        fi
        docker run --name shopping -p 7770:80 -d shopping_final_0712
        log_info "等待 Shopping 启动 (60秒)..."
        sleep 60
        docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://${SERVER_HOSTNAME}:7770"
        docker exec shopping mysql -u magentouser -pMyPassword magentodb -e 'UPDATE core_config_data SET value="http://'"${SERVER_HOSTNAME}"':7770/" WHERE path = "web/secure/base_url";'
        docker exec shopping /var/www/magento2/bin/magento cache:flush

        # 禁用 re-indexing (防止容器内 cron job 导致性能问题)
        for idx in catalogrule_product catalogrule_rule catalogsearch_fulltext catalog_category_product customer_grid design_config_grid inventory catalog_product_category catalog_product_attribute catalog_product_price cataloginventory_stock; do
            docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule $idx
        done
        log_info "Shopping 启动完成: http://${SERVER_HOSTNAME}:7770"
    fi

    # ---- 3d: Classifieds 网站 (端口 9980) ----
    log_info "=== 设置 Classifieds 网站 ==="
    if docker ps --format '{{.Names}}' | grep -q "classifieds"; then
        log_warn "Classifieds 容器已运行，跳过"
    else
        CLASSIFIEDS_DIR="$VWA_DIR/classifieds_docker_compose"
        if [ ! -d "$CLASSIFIEDS_DIR" ]; then
            cd "$VWA_DIR"
            unzip -q "$DOWNLOAD_DIR/classifieds_docker_compose.zip"
        fi
        cd "$CLASSIFIEDS_DIR"
        log_info "启动 Classifieds..."
        docker compose up --build -d
        sleep 10
        docker exec classifieds_db mysql -u root -ppassword osclass -e 'source docker-entrypoint-initdb.d/osclass_craigslist.sql' 2>/dev/null || true
        log_info "Classifieds 启动完成: http://${SERVER_HOSTNAME}:9980"
    fi

    # ---- 3e: Wikipedia 网站 (端口 8888, 可选) ----
    if [ "${SKIP_WIKIPEDIA:-1}" = "0" ]; then
        log_info "=== 设置 Wikipedia 网站 ==="
        if docker ps --format '{{.Names}}' | grep -q "wikipedia"; then
            log_warn "Wikipedia 容器已运行，跳过"
        else
            WIKI_DATA_DIR="$VWA_DIR/wiki_data"
            docker run -d --name=wikipedia --volume="$WIKI_DATA_DIR":/data -p 8888:80 ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2022-05.zim
            log_info "Wikipedia 启动完成: http://${SERVER_HOSTNAME}:8888"
        fi
    else
        log_warn "跳过 Wikipedia (我们选的 100 个评估 task 不需要 Wikipedia)"
    fi

    # ---- 3f: Homepage 网站 (端口 4399) ----
    log_info "=== 设置 Homepage ==="
    cd "$VWA_DIR/environment_docker/webarena-homepage"
    # 替换 hostname
    sed -i "s|<your-server-hostname>|${SERVER_HOSTNAME}|g" templates/index.html 2>/dev/null || true
    if ss -tlnp | grep -q ":4399 "; then
        log_warn "Homepage (端口 4399) 已在运行，跳过"
    else
        pip install flask -q 2>/dev/null
        nohup flask run --host=0.0.0.0 --port=4399 > /tmp/homepage.log 2>&1 &
        log_info "Homepage 启动完成: http://${SERVER_HOSTNAME}:4399"
    fi

    cd "$VWA_DIR"
    log_info "Step 3 完成 ✓"
    log_info "请访问以下地址验证网站是否正常:"
    log_info "  Classifieds: http://${SERVER_HOSTNAME}:9980"
    log_info "  Shopping:    http://${SERVER_HOSTNAME}:7770"
    log_info "  Reddit:      http://${SERVER_HOSTNAME}:9999"
    [ "${SKIP_WIKIPEDIA:-1}" = "0" ] && log_info "  Wikipedia:   http://${SERVER_HOSTNAME}:8888"
    log_info "  Homepage:    http://${SERVER_HOSTNAME}:4399"
}

###############################################################################
# Step 4: 环境变量 + 生成任务配置 + 登录 Cookie
###############################################################################
step4_configure() {
    log_step "Step 4: 配置环境变量 + 生成任务 + 登录 Cookie"
    cd "$VWA_DIR"

    ensure_conda
    conda activate "${CONDA_ENV_NAME}"

    # 设置环境变量
    export DATASET=visualwebarena
    export CLASSIFIEDS="http://${SERVER_HOSTNAME}:9980"
    export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
    export SHOPPING="http://${SERVER_HOSTNAME}:7770"
    export REDDIT="http://${SERVER_HOSTNAME}:9999"
    export WIKIPEDIA="http://${SERVER_HOSTNAME}:8888"
    export HOMEPAGE="http://${SERVER_HOSTNAME}:4399"

    log_info "环境变量已设置:"
    log_info "  DATASET=$DATASET"
    log_info "  CLASSIFIEDS=$CLASSIFIEDS"
    log_info "  SHOPPING=$SHOPPING"
    log_info "  REDDIT=$REDDIT"
    log_info "  WIKIPEDIA=$WIKIPEDIA"
    log_info "  HOMEPAGE=$HOMEPAGE"

    # 生成测试数据配置
    log_info "生成任务配置文件..."
    python3 scripts/generate_test_data.py
    log_info "配置文件已生成到 config_files/vwa/"

    # 获取登录 Cookie
    log_info "获取登录 Cookie..."
    mkdir -p .auth
    bash prepare.sh
    log_info "Cookie 已保存到 .auth/"

    # 写一个环境变量文件，方便后续 source
    cat > "$VWA_DIR/env_vwa.sh" << ENVEOF
#!/bin/bash
# VisualWebArena 环境变量 - source 此文件即可
export DATASET=visualwebarena
export CLASSIFIEDS="http://${SERVER_HOSTNAME}:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
export SHOPPING="http://${SERVER_HOSTNAME}:7770"
export REDDIT="http://${SERVER_HOSTNAME}:9999"
export WIKIPEDIA="http://${SERVER_HOSTNAME}:8888"
export HOMEPAGE="http://${SERVER_HOSTNAME}:4399"
export OPENAI_API_KEY="${VLLM_API_KEY}"
export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
ENVEOF
    chmod +x "$VWA_DIR/env_vwa.sh"
    log_info "环境变量文件已写入: env_vwa.sh (可通过 source env_vwa.sh 加载)"

    log_info "Step 4 完成 ✓"
}

###############################################################################
# Step 5: (可选) 在 tmux 中启动 vLLM 模型服务
#
# 注意: 对抗评估时建议直接使用 run_adversarial_eval.sh，它会自动管理 vLLM。
#       本 step 适用于手动启动单个 vLLM 做调试/单任务测试。
###############################################################################
step5_vllm() {
    log_step "Step 5: 在 tmux 中启动 vLLM 模型服务"

    if [ -z "$MODEL_PATH" ]; then
        log_error "请设置 MODEL_PATH 环境变量，例如:"
        log_error "  MODEL_PATH=/path/to/model ./setup_vwa.sh step5"
        return 1
    fi

    ensure_conda

    # 检查端口是否已占用
    local URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
    if curl -s "${URL}/models" 2>/dev/null | grep -q '"data"'; then
        local current=$(curl -s "${URL}/models" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else '')" 2>/dev/null || echo "")
        log_warn "vLLM 已在 ${VLLM_HOST}:${VLLM_PORT} 运行，当前模型: ${current}"
        log_warn "如需切换模型，请先: tmux kill-session -t vllm_server"
        return 0
    fi

    SESSION_NAME="vllm_server"
    log_info "启动 vLLM 服务..."
    log_info "  模型:     ${MODEL_PATH}"
    log_info "  端口:     ${VLLM_PORT}"
    log_info "  TP size:  ${VLLM_TP_SIZE}"
    log_info "  Max len:  ${VLLM_MAX_MODEL_LEN}"
    log_info "  tmux session: ${SESSION_NAME}"

    # Kill existing session
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    sleep 1

    # Find conda.sh
    CONDA_SH=""
    for p in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
        [ -f "$p" ] && CONDA_SH="$p" && break
    done

    tmux new-session -d -s "$SESSION_NAME"
    tmux send-keys -t "$SESSION_NAME" \
        "source ${CONDA_SH} && conda activate ${VLLM_CONDA_ENV} && vllm serve ${MODEL_PATH} \
         --port ${VLLM_PORT} \
         --tensor-parallel-size ${VLLM_TP_SIZE} \
         --max-model-len ${VLLM_MAX_MODEL_LEN} \
         --gpu-memory-utilization ${VLLM_GPU_MEM_UTIL} \
         --dtype auto \
         --trust-remote-code \
         --limit-mm-per-prompt '{\"image\": 5}' \
         2>&1 | tee /tmp/vllm_${SESSION_NAME}.log" C-m

    log_info "vLLM 启动命令已发送到 tmux session '${SESSION_NAME}'"
    log_info "等待 vLLM 就绪 (可能需要几分钟)..."

    for i in $(seq 1 120); do
        if curl -s "${URL}/models" 2>/dev/null | grep -q '"data"'; then
            log_info "vLLM 启动成功 ✓"
            local model=$(curl -s "${URL}/models" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null)
            log_info "  serving: ${model}"
            return 0
        fi
        echo -n "."
        sleep 5
    done

    log_error "vLLM 启动超时 (${i}*5s)，请检查:"
    log_error "  tmux attach -t ${SESSION_NAME}"
    log_error "  cat /tmp/vllm_${SESSION_NAME}.log"
    return 1
}

###############################################################################
# 主入口
###############################################################################
print_usage() {
    echo "用法: $0 {step1|step2|step3|step4|step5|all}"
    echo ""
    echo "  step1  - Python 环境安装 (conda env 'vwa' + 依赖)"
    echo "  step2  - 代码 patch (支持自定义多模态模型 + Gemma 角色交替)"
    echo "  step3  - Docker 网站环境搭建 (Classifieds, Shopping, Reddit, Homepage)"
    echo "  step4  - 环境变量 + 生成任务配置 + 登录 Cookie"
    echo "  step5  - (可选) 在 tmux 中启动单个 vLLM 服务"
    echo "  all    - 运行 step1 + step2 + step3 + step4"
    echo ""
    echo "配置完成后运行评估:"
    echo ""
    echo "  # 对抗评估 (100 tasks, 8 workers 并行):"
    echo "  bash run_adversarial_eval.sh --agent /path/to/agent_model --attacker /path/to/attacker_model"
    echo ""
    echo "  # 无攻击的 clean 评估:"
    echo "  bash run_adversarial_eval.sh --agent /path/to/model --no-attacker"
    echo ""
    echo "  # 查看所有选项:"
    echo "  head -30 run_adversarial_eval.sh"
    echo ""
    echo "关键配置变量 (通过环境变量或修改脚本顶部):"
    echo "  SERVER_HOSTNAME  - Docker 网站服务器地址 (默认: localhost)"
    echo "  VLLM_TP_SIZE     - Tensor Parallel size (默认: 4)"
    echo "  VLLM_CONDA_ENV   - 运行 vLLM 的 conda 环境 (默认: spag)"
    echo ""
    echo "示例:"
    echo "  # 在新机器上一键配置 (Docker 在本机):"
    echo "  SERVER_HOSTNAME=\$(hostname -I | awk '{print \$1}') ./setup_vwa.sh all"
    echo ""
    echo "  # Docker 在远程服务器:"
    echo "  SERVER_HOSTNAME=10.0.0.5 ./setup_vwa.sh all"
}

case "${1:-}" in
    step1)
        step1_python_env
        ;;
    step2)
        step2_code_patch
        ;;
    step3)
        step3_docker_setup
        ;;
    step4)
        step4_configure
        ;;
    step5)
        step5_vllm
        ;;
    all)
        step1_python_env
        step2_code_patch
        step3_docker_setup
        step4_configure
        log_info ""
        log_info "====== 配置完成! ======"
        log_info ""
        log_info "现在可以运行对抗评估了:"
        log_info ""
        log_info "  # 在 tmux 中运行 (推荐):"
        log_info "  tmux new -s eval"
        log_info "  bash run_adversarial_eval.sh --agent /path/to/agent --attacker /path/to/attacker"
        log_info ""
        log_info "  # 或无攻击 clean 评估:"
        log_info "  bash run_adversarial_eval.sh --agent /path/to/agent --no-attacker"
        log_info ""
        log_info "run_adversarial_eval.sh 会自动在 tmux 中启动 vLLM 服务器。"
        ;;
    *)
        print_usage
        ;;
esac
