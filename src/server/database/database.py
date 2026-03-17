from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    LargeBinary,
    ForeignKey,
    Index,
    UniqueConstraint,
    PrimaryKeyConstraint,
    text
)

from sqlalchemy.orm import sessionmaker, declarative_base, relationship
import os

DATABASE_URL = os.environ.get( "DATABASE_URL", "postgresql+psycopg2://postgres:example@localhost:5432/crawldb" )

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class DataType(Base):
    __tablename__ = "data_type"
    __table_args__ = ({"schema": "public"},)

    code = Column(String(20), primary_key=True)


class PageType(Base):
    __tablename__ = "page_type"
    __table_args__ = ({"schema": "public"},)

    code = Column(String(20), primary_key=True)


class Site(Base):
    __tablename__ = "site"
    __table_args__ = ({"schema": "public"},)

    id = Column(Integer, primary_key=True)
    domain = Column(String(500))
    robots_content = Column(Text)
    sitemap_content = Column(Text)


class Page(Base):
    __tablename__ = "page"
    __table_args__ = (
        UniqueConstraint("url", name="unq_url_idx"),
        Index("idx_page_site_id", "site_id"),
        Index("idx_page_page_type_code", "page_type_code"),
        {"schema": "public"},
    )

    id = Column(Integer, primary_key=True)
    site_id = Column(Integer, ForeignKey("public.site.id"))
    page_type_code = Column(String(20), ForeignKey("public.page_type.code"))
    url = Column(String(3000))
    html_content = Column(Text)
    http_status_code = Column(Integer)
    accessed_time = Column(String)

    site = relationship("Site", backref="pages")


class PageData(Base):
    __tablename__ = "page_data"
    __table_args__ = (
        Index("idx_page_data_page_id", "page_id"),
        Index("idx_page_data_data_type_code", "data_type_code"),
        {"schema": "public"},
    )

    id = Column(Integer, primary_key=True)
    page_id = Column(Integer, ForeignKey("public.page.id"))
    data_type_code = Column(String(20), ForeignKey("public.data_type.code"))
    data = Column(LargeBinary)

    page = relationship("Page", backref="data_items")


class Image(Base):
    __tablename__ = "image"
    __table_args__ = (Index("idx_image_page_id", "page_id"), {"schema": "public"})

    id = Column(Integer, primary_key=True)
    page_id = Column(Integer, ForeignKey("public.page.id"))
    filename = Column(String(255))
    content_type = Column(String(50))
    data = Column(LargeBinary)
    accessed_time = Column(String)


class Link(Base):
    __tablename__ = "link"
    __table_args__ = (
        PrimaryKeyConstraint("from_page", "to_page"),
        Index("idx_link_from_page", "from_page"),
        Index("idx_link_to_page", "to_page"),
        {"schema": "public"},
    )

    from_page = Column(Integer, ForeignKey("public.page.id"))
    to_page = Column(Integer, ForeignKey("public.page.id"))


def init():
    print("Initializing database...")

    try:
        with engine.begin() as conn:
            print("Creating schema...")
            Base.metadata.create_all(bind=conn.engine)

            conn.execute(
                text(
                    "INSERT INTO public.data_type(code) VALUES (:v) ON CONFLICT (code) DO NOTHING"
                ),
                [{"v": v} for v in ("PDF", "DOC", "DOCX", "PPT", "PPTX")],
            )
            conn.execute(
                text(
                    "INSERT INTO public.page_type(code) VALUES (:v) ON CONFLICT (code) DO NOTHING"
                ),
                [{"v": v} for v in ("HTML", "BINARY", "DUPLICATE", "FRONTIER")],
            )

            print("Initialized.")
    except Exception as e:
        print(e)