# Modal Authentication for Arc CLI

Arc CLI uses Modal for scalable scenario execution. Here are your authentication options:

## Option 1: Use Your Own Modal Account (Recommended for Production)

```bash
# One-time setup
modal token new

# Arc will automatically detect your Modal configuration
arc run agent.yaml
```

**Benefits:**
- Full control over compute resources
- Direct billing to your Modal account
- No shared resource limits
- Production-ready

## Option 2: Arc Demo Mode (For Testing Only)

Arc can provide demo credentials for evaluation purposes:

```bash
# Set Arc demo credentials (provided by Arc team)
export ARC_MODAL_TOKEN_ID="<provided-token>"
export ARC_MODAL_TOKEN_SECRET="<provided-secret>"

# Run with demo credentials
arc run agent.yaml --scenarios 10
```

**Limitations:**
- Shared resource pool with rate limits
- Maximum 50 scenarios per run
- For evaluation only, not production use

## Option 3: Deploy Arc Modal App to Your Workspace

For enterprise users who want centralized billing:

```bash
# Deploy Arc's Modal app to your workspace
cd arc-cli
modal deploy arc/sandbox/engine/simulator.py

# Set your workspace
export MODAL_WORKSPACE="your-company"

# Run Arc using your deployed app
arc run agent.yaml
```

## Security Best Practices

1. **Never commit credentials** to version control
2. **Use environment variables** or Modal secrets for CI/CD
3. **Rotate demo credentials** regularly
4. **Monitor usage** through Modal dashboard

## Troubleshooting

If you see "Modal not configured", try:

1. Check authentication:
   ```bash
   modal token status
   ```

2. Verify environment variables:
   ```bash
   echo $MODAL_TOKEN_ID
   echo $ARC_MODAL_TOKEN_ID
   ```

3. Test connection:
   ```bash
   modal run --help
   ```

## Setting Up for Teams

For teams using Arc, we recommend:

1. Each developer uses their own Modal account for development
2. Shared Modal workspace for staging/production
3. CI/CD uses Modal secrets:
   ```yaml
   # .github/workflows/arc-test.yml
   - name: Run Arc Tests
     env:
       MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
       MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
     run: |
       arc run config.yaml --scenarios 100
   ```

## Demo Request

To request Arc demo credentials:
- Contact: support@arc-eval.com
- Include: Company name, use case, expected volume

Demo credentials are valid for 30 days and include up to 1,000 scenario executions.