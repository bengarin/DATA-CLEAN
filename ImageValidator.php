<?php

declare(strict_types=1);

require_once __DIR__ . '/ApiClient.php';

/**
 * Validates the incoming upload locally (presence, size, real image type),
 * forwards it to the external Image Quality Validation API via ApiClient,
 * and guarantees the temporary file is removed — nothing is ever persisted.
 */
final class ImageValidator
{
    private ApiClient $apiClient;

    /** @param array<string, mixed> $config */
    public function __construct(private readonly array $config)
    {
        $this->apiClient = new ApiClient(
            $config['external_api']['url'],
            $config['external_api']['api_key'],
            $config['external_api']['field_name'],
            $config['external_api']['timeout'],
            $config['external_api']['connect_timeout'],
        );
    }

    /**
     * @param array<string, mixed>|null $file One entry from $_FILES, or null if absent.
     * @return array{httpStatus: int, body: array<string, mixed>}
     */
    public function handleUpload(?array $file): array
    {
        try {
            $this->assertValidUpload($file);

            $mimeType = $this->detectMimeType($file['tmp_name']);
            $this->assertAllowedMimeType($mimeType);

            $result = $this->apiClient->validateImage($file['tmp_name'], $file['name'], $mimeType);

            return ['httpStatus' => 200, 'body' => $result];
        } catch (InvalidArgumentException $e) {
            return ['httpStatus' => 400, 'body' => ['status' => 'error', 'message' => $e->getMessage()]];
        } catch (RuntimeException $e) {
            $status = $e->getCode() >= 400 && $e->getCode() < 600 ? $e->getCode() : 502;
            return ['httpStatus' => $status, 'body' => ['status' => 'error', 'message' => $e->getMessage()]];
        } finally {
            // Always clean up — the API must never leave the uploaded file on disk.
            if (isset($file['tmp_name']) && is_string($file['tmp_name']) && is_uploaded_file($file['tmp_name'])) {
                @unlink($file['tmp_name']);
            }
        }
    }

    /** @param array<string, mixed>|null $file */
    private function assertValidUpload(?array $file): void
    {
        $fieldName = $this->config['upload']['field_name'];

        if ($file === null) {
            throw new InvalidArgumentException("No file uploaded. Expected multipart field '{$fieldName}'.");
        }

        $errorCode = (int) ($file['error'] ?? UPLOAD_ERR_NO_FILE);
        if ($errorCode !== UPLOAD_ERR_OK) {
            throw new InvalidArgumentException('Upload error: ' . $this->uploadErrorMessage($errorCode));
        }

        if (!is_uploaded_file($file['tmp_name'])) {
            throw new InvalidArgumentException('Invalid upload.');
        }

        if ((int) $file['size'] <= 0) {
            throw new InvalidArgumentException('Uploaded file is empty.');
        }

        $maxSize = (int) $this->config['upload']['max_size_bytes'];
        if ((int) $file['size'] > $maxSize) {
            $maxMb = round($maxSize / 1024 / 1024, 1);
            throw new InvalidArgumentException("File exceeds the maximum allowed size of {$maxMb}MB.");
        }

        if (@getimagesize($file['tmp_name']) === false) {
            throw new InvalidArgumentException('Uploaded file is not a valid image.');
        }
    }

    private function detectMimeType(string $path): string
    {
        $finfo = finfo_open(FILEINFO_MIME_TYPE);
        $mime = $finfo !== false ? finfo_file($finfo, $path) : false;
        if ($finfo !== false) {
            finfo_close($finfo);
        }

        return $mime !== false ? $mime : 'application/octet-stream';
    }

    private function assertAllowedMimeType(string $mimeType): void
    {
        $allowed = $this->config['upload']['allowed_mime_types'];
        if (!in_array($mimeType, $allowed, true)) {
            throw new InvalidArgumentException(
                "Unsupported image type '{$mimeType}'. Allowed types: " . implode(', ', $allowed) . '.'
            );
        }
    }

    private function uploadErrorMessage(int $code): string
    {
        return match ($code) {
            UPLOAD_ERR_INI_SIZE, UPLOAD_ERR_FORM_SIZE => 'File exceeds the server upload size limit.',
            UPLOAD_ERR_PARTIAL => 'File was only partially uploaded.',
            UPLOAD_ERR_NO_FILE => 'No file was uploaded.',
            UPLOAD_ERR_NO_TMP_DIR => 'Missing temporary upload folder on server.',
            UPLOAD_ERR_CANT_WRITE => 'Failed to write file to disk.',
            UPLOAD_ERR_EXTENSION => 'Upload stopped by a PHP extension.',
            default => 'Unknown upload error.',
        };
    }
}
