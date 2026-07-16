<?php

declare(strict_types=1);

/**
 * Thin cURL client responsible for forwarding an image file to the
 * external Image Quality Validation API and returning its decoded JSON.
 */
final class ApiClient
{
    public function __construct(
        private readonly string $url,
        private readonly string $apiKey,
        private readonly string $fieldName,
        private readonly int $timeout,
        private readonly int $connectTimeout,
    ) {
    }

    /**
     * Sends the file at $filePath to the external API as multipart/form-data.
     *
     * @return array<string, mixed> Decoded JSON body returned by the external API.
     *
     * @throws RuntimeException on network failure, non-2xx response, or invalid JSON.
     */
    public function validateImage(string $filePath, string $originalName, string $mimeType): array
    {
        if ($this->url === '') {
            throw new RuntimeException('External API URL is not configured.', 500);
        }

        $curlFile = new CURLFile($filePath, $mimeType, $originalName);

        $headers = ['Accept: application/json'];
        if ($this->apiKey !== '') {
            $headers[] = "Authorization: Bearer {$this->apiKey}";
        }

        $ch = curl_init();
        curl_setopt_array($ch, [
            CURLOPT_URL => $this->url,
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => [$this->fieldName => $curlFile],
            CURLOPT_HTTPHEADER => $headers,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT => $this->timeout,
            CURLOPT_CONNECTTIMEOUT => $this->connectTimeout,
            CURLOPT_FAILONERROR => false,
        ]);

        $response = curl_exec($ch);
        $curlErrno = curl_errno($ch);
        $curlError = curl_error($ch);
        $httpCode = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        if ($curlErrno !== 0) {
            throw new RuntimeException("Could not reach the external validation API: {$curlError}", 502);
        }

        if ($httpCode < 200 || $httpCode >= 300) {
            throw new RuntimeException("External validation API returned HTTP {$httpCode}.", 502);
        }

        $decoded = json_decode((string) $response, true);
        if (!is_array($decoded)) {
            throw new RuntimeException('External validation API returned a non-JSON response.', 502);
        }

        return $decoded;
    }
}
