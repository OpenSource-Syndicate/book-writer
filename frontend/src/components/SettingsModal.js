import { useState, useEffect } from "react";
import axios from "axios";
import { API } from "../App";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Settings, Key, Zap } from "lucide-react";
import { toast } from "sonner";

const SettingsModal = ({ open, onOpenChange }) => {
  const [settings, setSettings] = useState({
    use_custom_api: false,
    api_endpoint: "https://api.openai.com/v1",
    api_key: "",
    model_name: "gpt-4o",
    temperature: 0.7,
    max_tokens: 2000
  });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (open) {
      fetchSettings();
    }
  }, [open]);

  const fetchSettings = async () => {
    try {
      const response = await axios.get(`${API}/settings/api`);
      setSettings({
        ...response.data,
        api_key: "" // Don't show the key
      });
    } catch (error) {
      console.error("Error fetching settings:", error);
    }
  };

  const saveSettings = async () => {
    setLoading(true);
    try {
      await axios.post(`${API}/settings/api`, settings);
      toast.success("Settings saved successfully!");
      onOpenChange(false);
    } catch (error) {
      console.error("Error saving settings:", error);
      toast.error("Failed to save settings");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px]" data-testid="settings-dialog">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Settings className="w-5 h-5" />
            API Settings
          </DialogTitle>
          <DialogDescription>
            Configure your AI model settings for content generation
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Emergent Key Info */}
          <Alert className="bg-indigo-50 border-indigo-200" data-testid="emergent-key-info">
            <Zap className="w-4 h-4 text-indigo-600" />
            <AlertDescription className="text-sm text-indigo-900 ml-2">
              By default, we use the Emergent LLM key which works with OpenAI, Anthropic, and Google models.
              You can also use your own API endpoint and key below.
            </AlertDescription>
          </Alert>

          {/* Custom API Toggle */}
          <Card data-testid="custom-api-card">
            <CardHeader>
              <CardTitle className="text-base">Custom API Configuration</CardTitle>
              <CardDescription>Use your own OpenAI-compatible endpoint</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <Label htmlFor="custom-api" className="flex items-center gap-2">
                  <Key className="w-4 h-4" />
                  Use Custom API
                </Label>
                <Switch
                  id="custom-api"
                  data-testid="custom-api-toggle"
                  checked={settings.use_custom_api}
                  onCheckedChange={(checked) => setSettings({ ...settings, use_custom_api: checked })}
                />
              </div>

              {settings.use_custom_api && (
                <div className="space-y-4 pt-4 border-t">
                  <div className="space-y-2">
                    <Label htmlFor="api-endpoint">API Endpoint</Label>
                    <Input
                      id="api-endpoint"
                      data-testid="api-endpoint-input"
                      placeholder="https://api.openai.com/v1"
                      value={settings.api_endpoint}
                      onChange={(e) => setSettings({ ...settings, api_endpoint: e.target.value })}
                    />
                    <p className="text-xs text-gray-500">Base URL for OpenAI-compatible API</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="api-key">API Key</Label>
                    <Input
                      id="api-key"
                      data-testid="api-key-input"
                      type="password"
                      placeholder="sk-..."
                      value={settings.api_key}
                      onChange={(e) => setSettings({ ...settings, api_key: e.target.value })}
                    />
                    <p className="text-xs text-gray-500">Your API key will be stored securely</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="model-name">Model Name</Label>
                    <Input
                      id="model-name"
                      data-testid="model-name-input"
                      placeholder="gpt-4o"
                      value={settings.model_name}
                      onChange={(e) => setSettings({ ...settings, model_name: e.target.value })}
                    />
                    <p className="text-xs text-gray-500">e.g., gpt-4o, gpt-4-turbo, claude-3-opus</p>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="temperature">Temperature</Label>
                      <Input
                        id="temperature"
                        data-testid="temperature-input"
                        type="number"
                        step="0.1"
                        min="0"
                        max="2"
                        value={settings.temperature}
                        onChange={(e) => setSettings({ ...settings, temperature: parseFloat(e.target.value) })}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="max-tokens">Max Tokens</Label>
                      <Input
                        id="max-tokens"
                        data-testid="max-tokens-input"
                        type="number"
                        value={settings.max_tokens}
                        onChange={(e) => setSettings({ ...settings, max_tokens: parseInt(e.target.value) })}
                      />
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="flex justify-end gap-3">
          <Button variant="outline" onClick={() => onOpenChange(false)} data-testid="cancel-settings-button">
            Cancel
          </Button>
          <Button onClick={saveSettings} disabled={loading} className="bg-indigo-600 hover:bg-indigo-700" data-testid="save-settings-button">
            {loading ? "Saving..." : "Save Settings"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default SettingsModal;