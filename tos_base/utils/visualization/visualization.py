# visualization.py
import json
import os
import re
from pathlib import Path
from html import escape
from itertools import zip_longest
from typing import List, Dict, Optional
import copy
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from .html_templates import HTML_TEMPLATE, CSS_STYLES, JAVASCRIPT_CODE



class VisualizationHelper:
    """Helper class for data processing and HTML generation"""
    
    @staticmethod
    def dict_to_html(d: Dict) -> str:
        """Convert dictionary to HTML format with better styling"""
        if not d:
            return "<div class='empty-dict'>(none)</div>"
        
        html = "<div class='dict-container'>"
        for k, v in d.items():
            if isinstance(v, (int, float)):
                # Format numbers nicely
                if isinstance(v, float):
                    formatted_v = f"{v:.3f}" if v != int(v) else str(int(v))
                else:
                    formatted_v = str(v)
                html += f"<div class='dict-item'><span class='dict-key'>{escape(str(k))}:</span> <span class='dict-value number'>{formatted_v}</span></div>"
            elif isinstance(v, bool):
                # Color-code booleans
                color_class = "true" if v else "false"
                html += f"<div class='dict-item'><span class='dict-key'>{escape(str(k))}:</span> <span class='dict-value {color_class}'>{str(v)}</span></div>"
            elif isinstance(v, dict):
                # Handle nested dictionaries
                nested_html = VisualizationHelper.dict_to_html(v)
                html += f"<div class='dict-item nested'><span class='dict-key'>{escape(str(k))}:</span> <div class='dict-value nested-dict'>{nested_html}</div></div>"
            else:
                # String values with consistent styling
                html += f"<div class='dict-item'><span class='dict-key'>{escape(str(k))}:</span> <span class='dict-value string'>{escape(str(v))}</span></div>"
        html += "</div>"
        return html
    
    @staticmethod
    def extract_think_and_answer(text: str) -> tuple[str, str]:
        """Extract think and answer content from text using regex patterns"""
        think_pattern = r'<think>(.*?)</think>'
        answer_pattern = r'<answer>(.*?)</answer>'
        
        think_match = re.search(think_pattern, text, re.DOTALL)
        answer_match = re.search(answer_pattern, text, re.DOTALL)
        
        think_content = think_match.group(1).strip() if think_match else text
        answer_content = answer_match.group(1).strip() if answer_match else text
        
        return think_content, answer_content
    
    @staticmethod
    def create_infogain_plot(infogain_per_turn: List[float], config_name: str) -> str:
        """Create a line plot for information gain per turn and return as base64 image."""

        fig, ax = plt.subplots(figsize=(8, 4))
        turns = list(range(1, len(infogain_per_turn) + 1))
        ax.plot(turns, infogain_per_turn, marker='o', linewidth=2, markersize=4)
        ax.set_xlabel('Turn')
        ax.set_ylabel('Average Information Gain')
        ax.set_title(f'Average Information Gain per Turn - {config_name}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, len(infogain_per_turn) + 0.5)
        
        # Save plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close(fig)
        
        # Convert to base64
        plot_base64 = base64.b64encode(plot_data).decode('utf-8')
        return f"data:image/png;base64,{plot_base64}"








class HTMLGenerator:
    """Handles HTML generation for the visualization"""
    
    def __init__(self, data: Dict, output_html: str, show_images: bool = True):
        self.data = data
        self.output_html = output_html
        self.show_images = show_images
        self.out_dir = os.path.dirname(output_html)
        self.base = Path(output_html).stem
        
        # Extract data
        self.meta = data.get("meta_info", {})
        self.config_groups = data["config_groups"]
        self.total_groups = len(self.config_groups)
        self.overall = data.get("overall_performance", {})
        
        # Extract summary data
        self.exp_summary = data.get("exp_summary", {})
        self.eval_summary = data.get("eval_summary", {})
        self.cogmap_summary = data.get("cogmap_summary", {})
        
        # Calculate statistics
        self.samples_per_group = {
            gname: len(group["env_data"])
            for gname, group in self.config_groups.items()
        }
        self.total_samples = sum(self.samples_per_group.values())
        self.total_pages = 1 + self.total_samples  # page 0 = TOC
        
        # Build flat list
        self.flat = []
        for gname, group in self.config_groups.items():
            for sidx, entry in enumerate(group["env_data"]):
                self.flat.append((gname, sidx, entry))

    def generate_summary_section(self, f) -> None:
        """Generate summary section with overall and group performance"""
        f.write("<div class='summary-section'>\n")
        f.write("<h3>📊 Overall Performance Summary</h3>\n")
        
        # Overall exploration efficiency
        if self.exp_summary.get("overall_performance"):
            exp_overall = self.exp_summary["overall_performance"]
            f.write("<div class='summary-card exploration'>\n")
            f.write("<h4>🔍 Exploration Summary</h4>\n")
            f.write(VisualizationHelper.dict_to_html(exp_overall))
            f.write("</div>\n")
        
        # Overall evaluation performance
        if self.eval_summary.get("overall_performance"):
            eval_overall = self.eval_summary["overall_performance"]
            f.write("<div class='summary-card evaluation'>\n")
            f.write("<h4>✅ Evaluation Summary</h4>\n")
            f.write(VisualizationHelper.dict_to_html(eval_overall))
            f.write("</div>\n")
        
        f.write("</div>\n")

    def generate_config_summaries(self, f) -> None:
        """Generate summaries for each config group"""
        f.write("<div class='config-summaries'>\n")
        f.write("<h3>📋 Configuration Summaries</h3>\n")
        
        for gname, group in self.config_groups.items():
            f.write(f"<div class='config-summary'>\n")
            f.write(f"<h4>⚙️ {escape(gname)}</h4>\n")
            f.write(f"<div class='config-stats'>\n")
            f.write(f"<div class='stat-item'>📊 Samples: {len(group['env_data'])}</div>\n")
            
            # Group exploration performance
            if self.exp_summary.get("group_performance", {}).get(gname):
                exp_group = self.exp_summary["group_performance"][gname]
                f.write("<div class='group-metrics'>")
                f.write("<strong>Exploration:</strong>")
                
                # Create infogain plot if data is available
                infogain_per_turn = exp_group.get("infogain_per_turn", [])
                if infogain_per_turn:
                    plot_src = VisualizationHelper.create_infogain_plot(infogain_per_turn, gname)
                    if plot_src:
                        f.write("<div class='infogain-plot'>")
                        f.write(f"<img src='{plot_src}' alt='Information Gain per Turn' style='max-width: 100%; height: auto; margin: 10px 0;'>")
                        f.write("</div>")
                
                # Display exploration metrics but exclude the infogain_per_turn list
                exp_group_filtered = {k: v for k, v in exp_group.items() if k != "infogain_per_turn"}
                f.write(VisualizationHelper.dict_to_html(exp_group_filtered))
                f.write("</div>\n")
            
            # Group evaluation performance
            if self.eval_summary.get("group_performance", {}).get(gname):
                eval_group = self.eval_summary["group_performance"][gname]
                f.write("<div class='group-metrics'>")
                f.write("<strong>Evaluation:</strong>")
                f.write(VisualizationHelper.dict_to_html(eval_group))
                f.write("</div>\n")

            # Group cognitive map performance
            if self.cogmap_summary.get("group_performance", {}).get(gname):
                cogmap_group = self.cogmap_summary["group_performance"][gname]
                f.write("<div class='group-metrics'>")
                f.write("<strong>Cognitive Map:</strong>")
                f.write(VisualizationHelper.dict_to_html(cogmap_group))
                f.write("</div>\n")
            
            f.write("</div>\n")
            f.write("</div>\n")
        
        f.write("</div>\n")

    def generate_toc_page(self, f) -> None:
        """Generate table of contents page with summaries"""
        f.write("<section class='sample-page active' id='page0'>\n")
        f.write("<h2>📋 Dashboard Overview</h2>\n")
        
        # Add summary sections
        self.generate_summary_section(f)
        self.generate_config_summaries(f)
        
        f.write("<h3>📖 Sample Navigation</h3>\n")
        f.write("<ul>\n")
        running_page = 1
        
        for gname, group in self.config_groups.items():
            f.write(f"<li><strong>{escape(gname)}</strong>\n  <ul>\n")
            for idx in range(len(group["env_data"])):
                label = f"Sample {idx+1}"
                f.write(
                    f"    <li>"
                    f"<a href='#' onclick=\"showPage({running_page}, {self.total_pages});return false;\">"
                    f"{label}</a></li>\n"
                )
                running_page += 1
            f.write("  </ul>\n</li>\n")
        f.write("</ul>\n</section>\n")

    def generate_sample_page(self, f, page_idx: int, gname: str, sidx: int, entry: Dict) -> None:
        """Generate a single sample page"""
        f.write(f"<section class='sample-page' id='page{page_idx}'>\n")
        f.write(f"<h2>{escape(gname)} — Sample {sidx+1}</h2>\n")

        # Display initial room image if available
        if self.show_images and entry.get("initial_room_image"):
            img_name = entry["initial_room_image"]
            f.write(f"<img src='{img_name}' class='room' alt='Initial room state'>\n")

        # Environment config
        cfg = entry["env_info"]["config"]
        f.write("<div class='metrics'><strong>🔧 Environment Configuration</strong>")
        f.write(VisualizationHelper.dict_to_html(cfg))
        f.write("</div>\n")

        # Get environment turn logs directly
        env_turn_logs = entry.get("env_turn_logs", [])
        
        # Generate turns from env logs
        for t_idx, env_log in enumerate(env_turn_logs):
            f.write("<div class='turn-split'>\n")
            f.write(f"<h3>🔄 Turn {t_idx+1}</h3>\n")
            
            # Left side: conversation and metrics
            f.write("<div class='turn-left'>\n")
            
            # Display user message (environment observation)
            if env_log['user_message']:
                u_short = escape(env_log['user_message'][:300]).replace("\n", "<br>")
                u_full = escape(env_log['user_message']).replace("\n", "<br>")
                obs_id = f"obs_{page_idx}_{t_idx}"
                f.write(f"<div class='block user expandable' onclick='expandObservation(\"{obs_id}\")'><strong>👤 Environment Observation <span class='expand-hint'>(click to expand)</span></strong><br>{u_short}...</div>\n")
                f.write(f"<div id='{obs_id}' class='observation-full' style='display:none'>{u_full}</div>\n")
                # u_full = escape(env_log['user_message']).replace("\n", "<br>")
                # f.write(f"<div class='block user'><strong>👤 Environment Observation</strong><br>{u_full}</div>\n")
            
            think_content, answer_content = env_log['assistant_think_message'], env_log['assistant_parsed_message']
            # Display think content
            if think_content:
                if False: # len(think_content) > 300:  # Make expandable if long
                    think_short = escape(think_content[:300]).replace("\n", "<br>")
                    think_full = escape(think_content).replace("\n", "<br>")
                    think_id = f"think_{page_idx}_{t_idx}"
                    f.write(f"<div class='block think expandable' onclick='expandThinking(\"{think_id}\")'><strong>🤔 Assistant Thinking <span class='expand-hint'>(click to expand)</span></strong><br>{think_short}...</div>\n")
                    f.write(f"<div id='{think_id}' class='thinking-full' style='display:none'>{think_full}</div>\n")
                else:
                    think = escape(think_content).replace("\n", "<br>")
                    f.write(f"<div class='block think'><strong>🤔 Assistant Thinking</strong><br>{think}</div>\n")
            # Display answer content
            if answer_content:
                answer = escape(answer_content).replace("\n", "<br>")
                f.write(f"<div class='block answer'><strong>💬 Assistant Action</strong><br>{answer}</div>\n")
            
            # Display evaluation information if available
            if not env_log['is_exploration_phase'] and env_log['evaluation_log']:
                eval_log = env_log['evaluation_log']
                f.write("<div class='block evaluation'><strong>✅ Evaluation</strong>")
                
                details = {
                    **eval_log["evaluation_data"],
                    **eval_log.get("evaluation_info", {}),
                    "Correct": eval_log.get("is_correct")
                }
                
                f.write(VisualizationHelper.dict_to_html(details))
                f.write("</div>\n")

            # Display cognitive map response if available
            if env_log.get('cognitive_map_response'):
                response_content = env_log['cognitive_map_response']
                f.write("<div class='block cogmap-response'><strong>🗺️ Cognitive Map Response</strong>")
                response_html = escape(response_content).replace("\n", "<br>")
                f.write(f"<div class='response-content'>{response_html}</div>")
                f.write("</div>\n")
            def _fmt_xy(v):
                try:
                    return f"[{int(v[0])},{int(v[1])}]"
                except Exception:
                    try:
                        return f"[{round(float(v[0]),2)},{round(float(v[1]),2)}]"
                    except Exception:
                        return str(v)

            def _compact_from_objmap(objmap: Dict[str, Dict]) -> str:
                """Compact text for {name:{position:[x,y], facing:str, ...}} dict."""
                if not isinstance(objmap, dict) or not objmap:
                    return "(none)"
                lines = []
                for name, info in objmap.items():
                    pos = _fmt_xy(info.get("position", [0, 0]))
                    facing = info.get("facing", "unknown")
                    lines.append(f"{name}: {pos}, {facing}")
                return "<br>".join(lines) if lines else "(none)"

            def _compact_rooms(rooms: Dict) -> str:
                """Compact text for rooms dict: { '1': {...}, '2': {...} }."""
                if not isinstance(rooms, dict) or not rooms:
                    return "(none)"
                parts = []
                for rid in sorted(rooms.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
                    body = _compact_from_objmap(rooms[rid])
                    parts.append(f"<div class='room-chunk'><strong>Room {escape(str(rid))}</strong><br>{body}</div>")
                return "".join(parts)

            def _compact_gates(gates: Dict) -> str:
                """Compact gates dict: { gate_name: {'connects':[a,b]} }."""
                if not isinstance(gates, dict) or not gates:
                    return "(none)"
                lines = []
                for gname, ginfo in gates.items():
                    conn = ginfo.get("connects", [])
                    try:
                        conn = [int(x) for x in conn]
                    except Exception:
                        pass
                    lines.append(f"{gname}: connects {conn}")
                return "<br>".join(lines) if lines else "(none)"

            def _metrics_dict_from_log(log: Dict) -> Dict[str, Dict]:
                """
                Build a tidy metrics dict with hierarchical + flat fields + consistency.
                """
                out = {
                    "Global": log.get("global", {}),
                    "Local": log.get("local", {}),
                    "Rooms": log.get("rooms", {}),
                    "Gates": (log.get("gates") or {}),
                }

                # Flat metrics (back-compat)
                flat = {
                    "dir_sim": log.get("dir_sim"),
                    "facing_sim": log.get("facing_sim"),
                    "pos_sim": log.get("pos_sim"),
                    "overall_sim": log.get("overall_sim"),
                }
                flat = {k: v for k, v in flat.items() if v is not None}
                if flat:
                    out["Summary"] = flat

                # Consistency block (if present)
                cons = log.get("consistency") or {}
                if isinstance(cons, dict) and cons:
                    # normalize fields we know about
                    local_vs_global = cons.get("local_vs_global", {}) or {}
                    rvg = cons.get("rooms_vs_global", {}) or {}
                    rooms_avg = rvg.get("average", {}) or {}
                    rooms_per = rvg.get("per_room", {}) or {}

                    out["Local vs Global"] = local_vs_global
                    out["Rooms vs Global (avg)"] = rooms_avg,
                    out["Rooms vs Global (per_room)"] = rooms_per

                return out

            def _render_cogmap_metrics_vs_gt(f, log: Dict, title: str):
                """
                Render side-by-side:
                LEFT: metrics (hierarchical + flat)
                RIGHT: ground truth cognitive map (global/local/rooms/gates)
                """
                # Left = metrics
                metrics_block = _metrics_dict_from_log(log)

                # Right = ground truth
                gt_global = log.get("gt_global_cog", {})
                gt_local  = log.get("gt_local_cog", {})
                gt_rooms  = log.get("gt_rooms_cog", {})
                gt_gates  = log.get("gt_gates", {})

                gt_global_txt = _compact_from_objmap(gt_global)
                gt_local_txt  = _compact_from_objmap(gt_local)
                gt_rooms_txt  = _compact_rooms(gt_rooms)
                gt_gates_txt  = _compact_gates(gt_gates)

                f.write("<div class='block cogmap'><strong>🧠 Cognitive Map (" + escape(title) + ")</strong>")
                f.write("<div class='cogmap-compare'>")

                # LEFT: Metrics
                f.write("<div class='cogmap-box side metrics-box'>")
                f.write("<div class='cogmap-box-title'>Metrics</div>")
                f.write(VisualizationHelper.dict_to_html(metrics_block))
                f.write("</div>")

                # RIGHT: Ground Truth
                f.write("<div class='cogmap-box side groundtruth-box framed'>")
                f.write("<div class='cogmap-box-title'>Ground Truth</div>")

                f.write("<div class='cogmap-gt-section'><div class='cogmap-section-title'>Global</div>")
                f.write(f"<div class='cogmap-box-body'>{gt_global_txt}</div></div>")

                f.write("<div class='cogmap-gt-section'><div class='cogmap-section-title'>Local</div>")
                f.write(f"<div class='cogmap-box-body'>{gt_local_txt}</div></div>")

                f.write("<div class='cogmap-gt-section'><div class='cogmap-section-title'>Rooms</div>")
                f.write(f"<div class='cogmap-box-body'>{gt_rooms_txt}</div></div>")

                f.write("<div class='cogmap-gt-section'><div class='cogmap-section-title'>Gates</div>")
                f.write(f"<div class='cogmap-box-body'>{gt_gates_txt}</div></div>")

                f.write("</div>")  # end RIGHT
                f.write("</div>")  # end compare row
                f.write("</div>")  # end block



            if env_log.get('cogmap_log'):
                _render_cogmap_metrics_vs_gt(f, env_log['cogmap_log'], "update")
            # Display cognitive map information if available
                
            if env_log.get('cogmap_full_log'):
                _render_cogmap_metrics_vs_gt(f, env_log['cogmap_full_log'], "full")


            # Display turn metrics from env log
            metrics = {}
            if env_log['is_exploration_phase'] and env_log['exploration_log']:
                exp_log = env_log['exploration_log']
                metrics.update({
                    "node_coverage": exp_log.get('node_coverage'),
                    "edge_coverage": exp_log.get('edge_coverage'),
                    "step": exp_log.get('step'),
                    "action_counts": exp_log.get('action_counts'),
                    "information_gain": exp_log.get('information_gain')
                })
            
            # Add info from env log
            if env_log['info']:
                metrics.update(env_log['info'])
            
            f.write("<div class='metrics'><strong>📈 Turn Metrics</strong>")
            f.write(VisualizationHelper.dict_to_html(metrics))
            f.write("</div>\n")
            f.write("</div>\n")  # End turn-left
            
            # Right side: room and message images
            f.write("<div class='turn-right'>\n")
            if self.show_images:
                # previous image (initial if first turn)
                prev_img = env_turn_logs[t_idx-1].get('room_image') if t_idx > 0 else None
                if prev_img:
                    f.write(f"<figure><img src='{prev_img}' class='room-plot' alt='Previous state'><figcaption>State before Turn {t_idx+1}</figcaption></figure>\n")
                # current image
                curr_img = env_log.get('room_image')
                if curr_img:
                    f.write(f"<figure><img src='{curr_img}' class='room-plot' alt='Current state'><figcaption>State at Turn {t_idx+1}</figcaption></figure>\n")
            # f.write("</div>\n")
            
            # Display message images
            if self.show_images and 'message_images' in env_log:
                for img_idx, img_path in enumerate(env_log['message_images']):
                    if isinstance(img_path, str):  # It's a path
                        # Check if this is likely the instruction image (first image in first turn)
                        # if img_idx == 0 and t_idx == 0:
                        #     f.write(f"<figure><img src='{img_path}' class='room-plot' alt='Instruction Image'><figcaption>📋 Task Instructions</figcaption></figure>\n")
                        # else:
                        # obs_number = img_idx + 1 if t_idx > 0 or img_idx > 0 else img_idx + 2
                        f.write(f"<figure><img src='{img_path}' class='room-plot' alt='Environment image {img_idx + 1}'><figcaption>Observation {img_idx + 1}</figcaption></figure>\n")
            f.write("</div>\n")  # End turn-right
            
            f.write("</div>\n")  # End turn-split

        # Final metrics
        summary = entry.get("summary", {})
        f.write("<div class='metrics'><strong>📊 Sample Final Metrics</strong>")
        f.write(VisualizationHelper.dict_to_html(summary))
        f.write("</div>\n")

        f.write("</section>\n")

    def generate_html(self) -> str:
        """Generate the complete HTML file"""
        with open(self.output_html, "w") as f:
            # Write HTML header with CSS and JS
            f.write(HTML_TEMPLATE.format(
                model_name=escape(self.meta.get('model_name', 'Unknown Model')),
                total_pages=self.total_pages,
                css_styles=CSS_STYLES,
                javascript_code=JAVASCRIPT_CODE
            ))

            # Generate TOC page (now with summaries)
            self.generate_toc_page(f)

            # Generate sample pages
            for page_idx, (gname, sidx, entry) in enumerate(self.flat, start=1):
                self.generate_sample_page(f, page_idx, gname, sidx, entry)

            f.write("</body></html>")

        return self.output_html


class Visualization:
    """Main visualization class for JSON data"""
    
    def __init__(self, json_path: str, output_html: str, show_images: bool = True):
        self.json_path = json_path
        self.output_html = output_html
        self.show_images = show_images

    def load_data(self) -> Dict:
        """Load JSON data from file"""
        with open(self.json_path, "r") as f:
            return json.load(f)

    def visualize(self) -> str:
        """Main method to generate visualization"""
        data = self.load_data()
        generator = HTMLGenerator(data, self.output_html, self.show_images)
        return generator.generate_html()


def visualize_json(json_path: str, output_html: str, show_images: bool = True) -> str:
    viz = Visualization(json_path, output_html, show_images)
    return viz.visualize()


		