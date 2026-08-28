"""
Example script demonstrating AlignTune raw file loaders.

This script shows how to:
1. Load different document types (PDF, DOCX, Markdown)
2. Use chunking options
3. Load from directories with mixed file types
4. Access loaded datasets
"""

import logging
from pathlib import Path
from aligntune.data.loaders.pdf_loader import PDFLoader
from aligntune.data.loaders.docx_loader import DocxLoader
from aligntune.data.loaders.markdown_loader import MarkdownLoader
from aligntune.data.loaders.directory_loader import DirectoryLoader
from aligntune.data.loaders.resolver import LoaderResolver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_pdf_loading():
    """Example 1: Load and process PDF files."""
    logger.info("=" * 60)
    logger.info("Example 1: PDF Loading")
    logger.info("=" * 60)

    # Load a single PDF without chunking (page-level granularity)
    logger.info("\n1.1: Loading PDF without chunking (page-level)")
    pdf_path = "path/to/document.pdf"  # Replace with actual path
    loader = PDFLoader(pdf_path, chunk_size=None)
    # dataset = loader.load()
    # print(f"Loaded {len(dataset)} pages from {pdf_path}")
    # for i, example in enumerate(dataset.take(2)):
    #     print(f"\nPage {example['page']}: {example['text'][:100]}...")

    # Load a single PDF with chunking
    logger.info("\n1.2: Loading PDF with chunking (512 token chunks)")
    loader = PDFLoader(pdf_path, chunk_size=512, chunk_overlap=50)
    # dataset = loader.load()
    # print(f"Loaded {len(dataset)} chunks from {pdf_path}")
    # for i, example in enumerate(dataset.take(2)):
    #     print(f"\nChunk {i}: {example['text'][:100]}...")

    # Load all PDFs from a directory
    logger.info("\n1.3: Loading all PDFs from a directory")
    pdf_dir = "path/to/pdf/directory"  # Replace with actual path
    loader = PDFLoader(pdf_dir)
    # dataset = loader.load_directory()
    # print(f"Loaded {len(dataset)} pages from {len(set(dataset['source']))} PDF files")
    # print("Sources:", set(dataset['source']))


def example_docx_loading():
    """Example 2: Load and process DOCX files."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: DOCX Loading")
    logger.info("=" * 60)

    # Load a single DOCX preserving paragraph structure
    logger.info("\n2.1: Loading DOCX preserving paragraph structure")
    docx_path = "path/to/document.docx"  # Replace with actual path
    loader = DocxLoader(docx_path, chunk_size=None)
    # dataset = loader.load()
    # print(f"Loaded {len(dataset)} paragraphs from {docx_path}")
    # for i, example in enumerate(dataset.take(3)):
    #     print(f"\nParagraph {example['paragraph_idx']}: {example['text']}")

    # Load a single DOCX with chunking
    logger.info("\n2.2: Loading DOCX with chunking (256 token chunks)")
    loader = DocxLoader(docx_path, chunk_size=256, chunk_overlap=30)
    # dataset = loader.load()
    # print(f"Loaded {len(dataset)} chunks from {docx_path}")

    # Load all DOCX files from a directory
    logger.info("\n2.3: Loading all DOCX files from a directory")
    docx_dir = "path/to/docx/directory"  # Replace with actual path
    loader = DocxLoader(docx_dir)
    # dataset = loader.load_directory()
    # print(f"Loaded {len(dataset)} paragraphs from {len(set(dataset['source']))} DOCX files")


def example_markdown_loading():
    """Example 3: Load and process Markdown files."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Markdown Loading")
    logger.info("=" * 60)

    md_path = "path/to/document.md"  # Replace with actual path

    # Load markdown with paragraph-based chunking
    logger.info("\n3.1: Loading Markdown with paragraph-based chunking")
    loader = MarkdownLoader(md_path, chunk_by="paragraph", chunk_size=None)
    # dataset = loader.load()
    # print(f"Loaded {len(dataset)} paragraphs from {md_path}")
    # for i, example in enumerate(dataset.take(3)):
    #     print(f"\nSection: {example['heading']}")
    #     print(f"Text: {example['text'][:100]}...")

    # Load markdown with heading-based chunking
    logger.info("\n3.2: Loading Markdown with heading-based chunking")
    loader = MarkdownLoader(md_path, chunk_by="heading")
    # dataset = loader.load()
    # print(f"Loaded {len(dataset)} sections (by heading) from {md_path}")
    # for i, example in enumerate(dataset.take(3)):
    #     print(f"\nHeading: {example['heading']}")
    #     print(f"Section: {example['section']}")
    #     print(f"Text: {example['text'][:100]}...")

    # Load all markdown files with heading-based structure
    logger.info("\n3.3: Loading all Markdown files from a directory")
    md_dir = "path/to/markdown/directory"  # Replace with actual path
    loader = MarkdownLoader(md_dir, chunk_by="heading")
    # dataset = loader.load_directory()
    # print(f"Loaded {len(dataset)} sections from markdown files")
    # print("Unique headings:", set(dataset['heading']))


def example_directory_loading():
    """Example 4: Load mixed file types from a directory."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Directory Loading (Mixed File Types)")
    logger.info("=" * 60)

    data_dir = "path/to/mixed/files"  # Replace with actual path

    # Load all supported files from directory
    logger.info("\n4.1: Loading all supported file types from directory")
    loader = DirectoryLoader(data_dir)
    # result = loader.load_directory()
    # if isinstance(result, Dataset):
    #     print(f"Loaded {len(result)} total items from {data_dir}")
    #     print(f"Sources: {set(result['source'])}")
    # else:
    #     print(f"Loaded {len(result)} datasets (DatasetDict)")
    #     for name, dataset in result.items():
    #         print(f"  - {name}: {len(dataset)} items")

    # Load only markdown files
    logger.info("\n4.2: Loading only Markdown files from directory")
    loader = DirectoryLoader(data_dir, pattern="*.md")
    # result = loader.load_directory()
    # print(f"Loaded {len(result)} markdown items")

    # Load recursively from subdirectories
    logger.info("\n4.3: Recursively loading all files from directory and subdirectories")
    loader = DirectoryLoader(data_dir, recurse=True)
    # result = loader.load_directory()
    # print(f"Loaded from directory with recursive=True")

    # View supported formats
    logger.info("\n4.4: Supported file formats")
    formats = DirectoryLoader.supported_formats()
    print(f"Supported formats: {', '.join(formats)}")


def example_using_resolver():
    """Example 5: Use LoaderResolver for automatic format detection."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Using LoaderResolver")
    logger.info("=" * 60)

    # Auto-detect format based on file extension
    logger.info("\n5.1: Auto-detect loader by file extension")
    pdf_path = "path/to/document.pdf"  # Replace with actual path
    loader = LoaderResolver.resolve(pdf_path)
    print(f"Resolved loader type: {type(loader).__name__}")

    # Resolve with custom parameters
    logger.info("\n5.2: Resolve with custom parameters")
    loader = LoaderResolver.resolve(
        pdf_path,
        chunk_size=256,
        chunk_overlap=20
    )
    print(f"Loader configuration: chunk_size={loader.chunk_size}, "
          f"chunk_overlap={loader.chunk_overlap}")

    # Resolve directory
    logger.info("\n5.3: Resolve directory")
    data_dir = "path/to/directory"  # Replace with actual path
    loader = LoaderResolver.resolve(data_dir, pattern="*.md")
    print(f"Resolved loader type: {type(loader).__name__}")

    # Get loader by format name
    logger.info("\n5.4: Get loader by format name")
    pdf_loader_class = LoaderResolver.get_loader_by_format("pdf")
    print(f"PDF loader class: {pdf_loader_class.__name__}")

    # View all supported formats
    logger.info("\n5.5: View all supported formats")
    formats = LoaderResolver.supported_formats()
    print(f"Supported formats: {', '.join(formats)}")


def example_advanced_chunking():
    """Example 6: Advanced chunking options."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 6: Advanced Chunking Options")
    logger.info("=" * 60)

    pdf_path = "path/to/document.pdf"  # Replace with actual path

    # Large chunks with overlap
    logger.info("\n6.1: Large chunks (1024 tokens) with 256 token overlap")
    loader = PDFLoader(pdf_path, chunk_size=1024, chunk_overlap=256)
    # dataset = loader.load()
    # print(f"Loaded {len(dataset)} chunks")

    # Small chunks (good for retrieval)
    logger.info("\n6.2: Small chunks (128 tokens) for fine-grained retrieval")
    loader = PDFLoader(pdf_path, chunk_size=128, chunk_overlap=16)
    # dataset = loader.load()
    # print(f"Loaded {len(dataset)} chunks")

    # Custom chunking per file type
    logger.info("\n6.3: Different chunking for different file types")

    # PDF with specific chunking
    pdf_loader = PDFLoader(pdf_path, chunk_size=512, chunk_overlap=50)

    # Markdown with heading-based chunking
    md_path = "path/to/document.md"  # Replace with actual path
    md_loader = MarkdownLoader(md_path, chunk_by="heading")

    print("Configured loaders with custom chunking strategies")


def example_dataset_inspection():
    """Example 7: Inspect and work with loaded datasets."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 7: Dataset Inspection")
    logger.info("=" * 60)

    md_path = "path/to/document.md"  # Replace with actual path
    loader = MarkdownLoader(md_path)

    logger.info("\nLoading example markdown file...")
    # dataset = loader.load()
    # if dataset is not None and len(dataset) > 0:
    #     # View column names
    #     print(f"Dataset columns: {dataset.column_names}")
    #
    #     # View dataset info
    #     print(f"Dataset size: {len(dataset)} examples")
    #
    #     # View first few examples
    #     print("\nFirst 3 examples:")
    #     for example in dataset.take(3):
    #         print(f"\n  Source: {example['source']}")
    #         print(f"  Heading: {example['heading']}")
    #         print(f"  Section: {example['section']}")
    #         print(f"  Text: {example['text'][:80]}...")
    #
    #     # Convert to pandas for analysis
    #     df = dataset.to_pandas()
    #     print(f"\nDataFrame shape: {df.shape}")
    #     print(f"Unique headings: {df['heading'].nunique()}")


def print_usage_summary():
    """Print a summary of the loaders and their use cases."""
    logger.info("\n" + "=" * 60)
    logger.info("Loader Summary")
    logger.info("=" * 60)

    summary = """
PDFLoader:
  - Use case: Extract text from PDF documents
  - Granularity: Page-level (without chunking) or custom chunks
  - Metadata: source (filename), page (page number or -1 if chunked)
  - Parameters: chunk_size, chunk_overlap

DocxLoader:
  - Use case: Extract text from Microsoft Word documents
  - Granularity: Paragraph-level (without chunking) or custom chunks
  - Metadata: source (filename), paragraph_idx
  - Parameters: chunk_size, chunk_overlap

MarkdownLoader:
  - Use case: Extract text from Markdown files with structure awareness
  - Granularity: Paragraph or heading-based sections
  - Metadata: source (filename), heading, section
  - Parameters: chunk_by ("paragraph" or "heading"), chunk_size, chunk_overlap

DirectoryLoader:
  - Use case: Load multiple files of different types from a directory
  - Granularity: Auto-routes each file to appropriate loader
  - Metadata: Depends on file type
  - Parameters: pattern (glob), recurse (recursive search)

LoaderResolver:
  - Use case: Automatically detect and instantiate appropriate loader
  - Methods: resolve() (auto-detect), get_loader_by_format() (by name)
  - Supported formats: pdf, docx, markdown, json, csv, parquet, directory

Common Parameters:
  - chunk_size: Number of tokens per chunk (None to disable)
  - chunk_overlap: Number of tokens overlap between chunks
  - pattern: Glob pattern for file filtering
  - recurse: Whether to search subdirectories

Output Format:
All loaders return Hugging Face Dataset with consistent structure:
  - "text": The extracted text content
  - "source": Source filename
  - Additional metadata specific to loader type
    """
    print(summary)


if __name__ == "__main__":
    logger.info("\n" + "=" * 60)
    logger.info("AlignTune Raw File Loaders - Examples")
    logger.info("=" * 60)
    logger.info("\nNote: Replace 'path/to/...' with actual file/directory paths")

    # Run examples (comment out or replace paths as needed)
    print_usage_summary()

    # Uncomment to run specific examples:
    # example_pdf_loading()
    # example_docx_loading()
    # example_markdown_loading()
    # example_directory_loading()
    # example_using_resolver()
    # example_advanced_chunking()
    # example_dataset_inspection()

    logger.info("\n" + "=" * 60)
    logger.info("Examples completed!")
    logger.info("=" * 60)
