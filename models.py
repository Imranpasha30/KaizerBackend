from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class Job(Base):
    __tablename__ = "jobs"

    id           = Column(Integer, primary_key=True, index=True)
    status       = Column(String(20), default="pending")   # pending | running | done | failed
    platform     = Column(String(50))
    frame_layout = Column(String(50))
    video_name   = Column(String(255))
    output_dir   = Column(String(500), default="")
    log          = Column(Text, default="")
    error        = Column(Text, default="")
    created_at   = Column(DateTime(timezone=True), server_default=func.now())

    clips = relationship("Clip", back_populates="job", cascade="all, delete")


class Clip(Base):
    __tablename__ = "clips"

    id           = Column(Integer, primary_key=True, index=True)
    job_id       = Column(Integer, ForeignKey("jobs.id"))
    clip_index   = Column(Integer, default=0)
    filename     = Column(String(255), default="")
    file_path    = Column(String(500), default="")
    thumb_path   = Column(String(500), default="")
    image_path   = Column(String(500), default="")
    duration     = Column(Float, default=0)
    frame_type   = Column(String(50), default="")
    text         = Column(Text, default="")
    sentiment    = Column(String(50), default="")
    entities     = Column(Text, default="[]")    # JSON array
    card_params  = Column(Text, default="{}")    # JSON: font_size, text_color, font_file, card_style
    section_pct  = Column(Text, default="{}")    # JSON: {video, text, image}
    follow_params= Column(Text, default="{}")    # JSON: follow_text, bg_color, etc.
    meta         = Column(Text, default="{}")    # raw pipeline meta JSON

    job = relationship("Job", back_populates="clips")
