import * as pdfjsLib from 'pdfjs-dist';
import mammoth from 'mammoth';
import * as XLSX from 'xlsx';

// Configure PDF.js worker for Vite
if (typeof window !== 'undefined') {
  // Use unpkg CDN with version 5.4.394 (matches installed version)
  // For pdfjs-dist 5.x, the worker is at build/pdf.worker.min.mjs
  pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://unpkg.com/pdfjs-dist@5.4.394/build/pdf.worker.min.mjs';
}

/**
 * Extract text from a PDF file
 */
export const extractPdfText = async (fileUrl) => {
  try {
    console.log('Fetching PDF from URL:', fileUrl);
    const response = await fetch(fileUrl);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const arrayBuffer = await response.arrayBuffer();
    console.log('PDF loaded, size:', arrayBuffer.byteLength, 'bytes');
    
    const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
    const pdf = await loadingTask.promise;
    console.log('PDF parsed, pages:', pdf.numPages);
    
    let fullText = '';
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      const pageText = textContent.items.map((item) => item.str).join(' ');
      fullText += pageText + '\n\n';
    }
    
    const extractedText = fullText.trim();
    console.log('PDF text extracted, length:', extractedText.length);
    return extractedText;
  } catch (error) {
    console.error('Error extracting PDF text:', error);
    throw new Error(`Failed to extract text from PDF: ${error.message}`);
  }
};

/**
 * Extract text from a DOCX file
 */
export const extractDocxText = async (fileUrl) => {
  try {
    const response = await fetch(fileUrl);
    const arrayBuffer = await response.arrayBuffer();
    const result = await mammoth.extractRawText({ arrayBuffer });
    return result.value.trim();
  } catch (error) {
    console.error('Error extracting DOCX text:', error);
    throw new Error(`Failed to extract text from DOCX: ${error.message}`);
  }
};

/**
 * Extract text/data from an Excel file (XLSX/XLS/CSV)
 */
export const extractExcelText = async (fileUrl, fileName) => {
  try {
    const response = await fetch(fileUrl);
    const arrayBuffer = await response.arrayBuffer();
    const workbook = XLSX.read(arrayBuffer, { type: 'array' });
    
    let fullText = '';
    workbook.SheetNames.forEach((sheetName) => {
      const worksheet = workbook.Sheets[sheetName];
      const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1, defval: '' });
      
      fullText += `Sheet: ${sheetName}\n`;
      jsonData.forEach((row) => {
        if (Array.isArray(row) && row.length > 0) {
          const rowText = row.filter((cell) => cell !== null && cell !== '').join(' | ');
          if (rowText.trim()) {
            fullText += rowText + '\n';
          }
        }
      });
      fullText += '\n';
    });
    
    return fullText.trim();
  } catch (error) {
    console.error('Error extracting Excel text:', error);
    throw new Error(`Failed to extract data from Excel file: ${error.message}`);
  }
};

/**
 * Extract text from a TXT file
 */
export const extractTxtText = async (fileUrl) => {
  try {
    const response = await fetch(fileUrl);
    const text = await response.text();
    return text.trim();
  } catch (error) {
    console.error('Error extracting TXT text:', error);
    throw new Error(`Failed to extract text from TXT file: ${error.message}`);
  }
};

/**
 * Extract data from a JSON file
 */
export const extractJsonText = async (fileUrl) => {
  try {
    const response = await fetch(fileUrl);
    const json = await response.json();
    return JSON.stringify(json, null, 2);
  } catch (error) {
    console.error('Error extracting JSON text:', error);
    throw new Error(`Failed to extract data from JSON file: ${error.message}`);
  }
};

/**
 * Main function to extract text from any supported file type
 */
export const extractFileText = async (fileUrl, fileName, fileType) => {
  const ext = fileName.split('.').pop()?.toLowerCase() || '';
  
  // Determine file type
  if (ext === 'pdf' || fileType?.includes('pdf')) {
    return await extractPdfText(fileUrl);
  } else if (ext === 'docx' || ext === 'doc' || fileType?.includes('word')) {
    return await extractDocxText(fileUrl);
  } else if (ext === 'xlsx' || ext === 'xls' || ext === 'csv' || fileType?.includes('sheet') || fileType?.includes('csv')) {
    return await extractExcelText(fileUrl, fileName);
  } else if (ext === 'txt' || fileType?.includes('text')) {
    return await extractTxtText(fileUrl);
  } else if (ext === 'json' || fileType?.includes('json')) {
    return await extractJsonText(fileUrl);
  } else {
    throw new Error(`Unsupported file type: ${ext}`);
  }
};

