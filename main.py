
import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# Core dependencies for video analysis
import cv2
import numpy as np
from PIL import Image
import requests
from openai import OpenAI
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FrameAnalysis:
    """Structure for individual frame analysis results"""
    timestamp: float
    frame_number: int
    objects_detected: List[str]
    scene_description: str
    emotions_detected: List[str]
    text_extracted: str
    confidence_scores: Dict[str, float]

@dataclass
class VideoAnalysisResult:
    """Complete video analysis result structure"""
    video_id: str
    duration: float
    total_frames: int
    frame_analyses: List[FrameAnalysis]
    summary: str
    key_insights: List[str]
    topics: List[str]
    sentiment_analysis: Dict[str, float]
    transcript: Optional[str] = None

class VideoAnalysisAgent:
    """Main Video Analysis Agent with LLM integration and RAG capabilities"""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    async def extract_frames(self, video_path: str, interval: int = 30) -> List[np.ndarray]:
        """Extract frames from video at specified intervals"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % (fps * interval) == 0:  # Extract every 'interval' seconds
                    frames.append(frame)
                    
                frame_count += 1
                
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []
    
    def analyze_frame_with_vision(self, frame: np.ndarray, timestamp: float) -> FrameAnalysis:
        """Analyze individual frame using OpenAI Vision API"""
        try:
            # Convert frame to base64 for API
            _, buffer = cv2.imencode('.jpg', frame)
            import base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prompt for comprehensive frame analysis
            prompt = """
            Analyze this video frame and provide:
            1. Objects and people detected
            2. Scene description (setting, environment)
            3. Emotions or expressions visible
            4. Any text present in the frame
            5. Overall mood/atmosphere
            
            Format your response as JSON with keys: objects, scene, emotions, text, mood
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            # Parse response
            analysis_text = response.choices[0].message.content
            try:
                analysis_data = json.loads(analysis_text)
            except:
                # Fallback if JSON parsing fails
                analysis_data = {
                    "objects": [],
                    "scene": analysis_text[:200],
                    "emotions": [],
                    "text": "",
                    "mood": "neutral"
                }
            
            return FrameAnalysis(
                timestamp=timestamp,
                frame_number=len(self.frame_analyses) if hasattr(self, 'frame_analyses') else 0,
                objects_detected=analysis_data.get("objects", []),
                scene_description=analysis_data.get("scene", ""),
                emotions_detected=analysis_data.get("emotions", []),
                text_extracted=analysis_data.get("text", ""),
                confidence_scores={"overall": 0.8}  # Placeholder
            )
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return FrameAnalysis(timestamp, 0, [], "Analysis failed", [], "", {})
    
    def build_knowledge_base(self, frame_analyses: List[FrameAnalysis]) -> None:
        """Build RAG knowledge base from frame analyses"""
        try:
            documents = []
            for analysis in frame_analyses:
                content = f"""
                Timestamp: {analysis.timestamp}s
                Scene: {analysis.scene_description}
                Objects: {', '.join(analysis.objects_detected)}
                Emotions: {', '.join(analysis.emotions_detected)}
                Text: {analysis.text_extracted}
                """
                documents.append(Document(page_content=content, metadata={"timestamp": analysis.timestamp}))
            
            # Split and create vector store
            texts = self.text_splitter.split_documents(documents)
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            logger.info(f"Built knowledge base with {len(texts)} text chunks")
            
        except Exception as e:
            logger.error(f"Error building knowledge base: {e}")
    
    def query_video_content(self, query: str, k: int = 5) -> str:
        """Query video content using RAG"""
        if not self.vector_store:
            return "Knowledge base not available"
            
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            context = "\n".join([doc.page_content for doc in docs])
            
            prompt = f"""
            Based on the following video analysis data, answer the user's question:
            
            Context from video:
            {context}
            
            Question: {query}
            
            Provide a comprehensive answer based on the video content:
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error querying video content: {e}")
            return "Error processing query"
    
    async def analyze_video(self, video_path: str, video_id: str) -> VideoAnalysisResult:
        """Complete video analysis pipeline"""
        try:
            # Extract frames
            frames = await self.extract_frames(video_path)
            if not frames:
                raise ValueError("No frames extracted from video")
            
            # Analyze each frame
            frame_analyses = []
            for i, frame in enumerate(frames):
                timestamp = i * 30.0  # Assuming 30-second intervals
                analysis = self.analyze_frame_with_vision(frame, timestamp)
                frame_analyses.append(analysis)
                
                # Progress indicator
                if i % 5 == 0:
                    logger.info(f"Analyzed {i+1}/{len(frames)} frames")
            
            # Build knowledge base
            self.build_knowledge_base(frame_analyses)
            
            # Generate summary and insights
            summary = await self.generate_video_summary(frame_analyses)
            insights = await self.extract_key_insights(frame_analyses)
            topics = await self.identify_topics(frame_analyses)
            sentiment = await self.analyze_sentiment(frame_analyses)
            
            # Get video duration
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            return VideoAnalysisResult(
                video_id=video_id,
                duration=duration,
                total_frames=len(frames),
                frame_analyses=frame_analyses,
                summary=summary,
                key_insights=insights,
                topics=topics,
                sentiment_analysis=sentiment
            )
            
        except Exception as e:
            logger.error(f"Error in video analysis: {e}")
            raise
    
    async def generate_video_summary(self, analyses: List[FrameAnalysis]) -> str:
        """Generate comprehensive video summary"""
        scenes = [analysis.scene_description for analysis in analyses]
        objects = set()
        emotions = set()
        
        for analysis in analyses:
            objects.update(analysis.objects_detected)
            emotions.update(analysis.emotions_detected)
        
        prompt = f"""
        Generate a comprehensive summary of this video based on the frame analyses:
        
        Scenes observed: {'; '.join(scenes[:10])}...
        Objects detected: {', '.join(list(objects)[:20])}
        Emotions observed: {', '.join(list(emotions))}
        
        Provide a 3-paragraph summary covering:
        1. Overall content and setting
        2. Key activities and objects
        3. Emotional tone and atmosphere
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        
        return response.choices[0].message.content
    
    async def extract_key_insights(self, analyses: List[FrameAnalysis]) -> List[str]:
        """Extract key insights from video analysis"""
        # Implementation for insight extraction
        insights = [
            f"Video contains {len(analyses)} analyzed segments",
            f"Most common emotion: {self._most_common_emotion(analyses)}",
            f"Primary setting: {self._identify_primary_setting(analyses)}"
        ]
        return insights
    
    async def identify_topics(self, analyses: List[FrameAnalysis]) -> List[str]:
        """Identify main topics in the video"""
        all_content = " ".join([a.scene_description for a in analyses])
        
        prompt = f"""
        Based on this video content, identify the main topics (5-7 topics):
        {all_content[:2000]}
        
        Return only a list of topics, one per line.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        
        return [topic.strip() for topic in response.choices[0].message.content.split('\n') if topic.strip()]
    
    async def analyze_sentiment(self, analyses: List[FrameAnalysis]) -> Dict[str, float]:
        """Analyze overall sentiment of the video"""
        emotions = []
        for analysis in analyses:
            emotions.extend(analysis.emotions_detected)
        
        # Simple sentiment mapping (can be enhanced with proper sentiment analysis)
        positive_emotions = ['happy', 'joy', 'excited', 'pleased', 'satisfied']
        negative_emotions = ['sad', 'angry', 'frustrated', 'disappointed', 'upset']
        neutral_emotions = ['calm', 'neutral', 'focused', 'serious']
        
        positive_count = sum(1 for emotion in emotions if emotion.lower() in positive_emotions)
        negative_count = sum(1 for emotion in emotions if emotion.lower() in negative_emotions)
        neutral_count = sum(1 for emotion in emotions if emotion.lower() in neutral_emotions)
        
        total = max(positive_count + negative_count + neutral_count, 1)
        
        return {
            "positive": positive_count / total,
            "negative": negative_count / total,
            "neutral": neutral_count / total
        }
    
    def _most_common_emotion(self, analyses: List[FrameAnalysis]) -> str:
        """Helper to find most common emotion"""
        emotions = []
        for analysis in analyses:
            emotions.extend(analysis.emotions_detected)
        
        if not emotions:
            return "neutral"
        
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral"
    
    def _identify_primary_setting(self, analyses: List[FrameAnalysis]) -> str:
        """Helper to identify primary setting"""
        scenes = [analysis.scene_description for analysis in analyses]
        if not scenes:
            return "unknown"
        
        # Simple keyword-based setting identification
        indoor_keywords = ['room', 'office', 'kitchen', 'bedroom', 'indoor']
        outdoor_keywords = ['park', 'street', 'outdoor', 'garden', 'field']
        
        indoor_count = sum(1 for scene in scenes if any(keyword in scene.lower() for keyword in indoor_keywords))
        outdoor_count = sum(1 for scene in scenes if any(keyword in scene.lower() for keyword in outdoor_keywords))
        
        return "indoor" if indoor_count > outdoor_count else "outdoor" if outdoor_count > 0 else "mixed"

# Streamlit UI
def create_streamlit_ui():
    """Create Streamlit interface for the Video Analysis Agent"""
    st.set_page_config(page_title="Video Analysis Agent", page_icon="🎬", layout="wide")
    
    st.title("🎬 Video Analysis Agent")
    st.markdown("Upload a video and get comprehensive AI-powered analysis including object detection, scene understanding, and sentiment analysis.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
        
        if api_key:
            st.success("API Key configured!")
        else:
            st.warning("Please enter your OpenAI API key to continue")
    
    if api_key:
        # Initialize agent
        if 'agent' not in st.session_state:
            st.session_state.agent = VideoAnalysisAgent(api_key)
        
        # File upload
        uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_file is not None:
            # Save uploaded file
            video_path = f"temp_video_{uploaded_file.name}"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.video(video_path)
            
            # Analysis button
            if st.button("🔍 Analyze Video", type="primary"):
                with st.spinner("Analyzing video... This may take a few minutes."):
                    try:
                        # Run analysis
                        result = asyncio.run(
                            st.session_state.agent.analyze_video(video_path, uploaded_file.name)
                        )
                        
                        # Store result in session state
                        st.session_state.analysis_result = result
                        
                        st.success("Analysis complete!")
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
            
            # Display results if available
            if 'analysis_result' in st.session_state:
                result = st.session_state.analysis_result
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Summary", "🔍 Frame Analysis", "💭 Insights", "❓ Q&A", "📈 Metrics"])
                
                with tab1:
                    st.header("Video Summary")
                    st.write(result.summary)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Key Topics")
                        for topic in result.topics:
                            st.write(f"• {topic}")
                    
                    with col2:
                        st.subheader("Sentiment Analysis")
                        sentiment = result.sentiment_analysis
                        st.write(f"Positive: {sentiment['positive']:.2%}")
                        st.write(f"Negative: {sentiment['negative']:.2%}")
                        st.write(f"Neutral: {sentiment['neutral']:.2%}")
                
                with tab2:
                    st.header("Frame-by-Frame Analysis")
                    for i, analysis in enumerate(result.frame_analyses[:10]):  # Show first 10 frames
                        with st.expander(f"Frame {i+1} (Time: {analysis.timestamp}s)"):
                            st.write(f"**Scene:** {analysis.scene_description}")
                            st.write(f"**Objects:** {', '.join(analysis.objects_detected)}")
                            st.write(f"**Emotions:** {', '.join(analysis.emotions_detected)}")
                            if analysis.text_extracted:
                                st.write(f"**Text:** {analysis.text_extracted}")
                
                with tab3:
                    st.header("Key Insights")
                    for insight in result.key_insights:
                        st.write(f"• {insight}")
                
                with tab4:
                    st.header("Ask Questions About the Video")
                    query = st.text_input("What would you like to know about this video?")
                    if query and st.button("🔍 Search"):
                        with st.spinner("Searching video content..."):
                            answer = st.session_state.agent.query_video_content(query)
                            st.write("**Answer:**")
                            st.write(answer)
                
                with tab5:
                    st.header("Analysis Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Duration", f"{result.duration:.1f}s")
                    with col2:
                        st.metric("Frames Analyzed", result.total_frames)
                    with col3:
                        st.metric("Unique Objects", len(set(obj for frame in result.frame_analyses for obj in frame.objects_detected)))
                    with col4:
                        st.metric("Unique Emotions", len(set(emotion for frame in result.frame_analyses for emotion in frame.emotions_detected)))
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using OpenAI GPT-4 Vision, LangChain, and Streamlit")

if __name__ == "__main__":
    # Check if we're running in Streamlit
    try:
        import streamlit.runtime.scriptrunner as scriptrunner
        create_streamlit_ui()
    except:
        # Fallback for direct execution
        print("Video Analysis Agent")
        print("Run with: streamlit run main.py")
        print("Make sure to set your OpenAI API key in the Streamlit interface")
